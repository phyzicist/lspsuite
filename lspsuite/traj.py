# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 17:31:13 2016

Scott's implementation of pmovie sorting.
Adapted from code by Gregory Ngmirmang

Changelog:
2019-11-09 Brought into line with Python3 conventions, and generalized to work with 1D/2D/3D sims
2016-07-06 Updated hashing to use Gregory's. Also, now fills in particles with their initial conditions (rather than most recent).
2016-07-06 Added a pre-agreed-upon random shuffling to the fillgaps() step, so that contiguous blocks in the HDF5 contain random particles

Example usage.
1. Write a script, "MyScript.py", with contents such as:
    from pmov.traj import mpitraj 
    p4dir = "/fs/scratch/PAS1066/GroupShare/Generation_904_11_7_2019/904/0"
    mpitraj(p4dir)
    
2. Run the script in parallel from command line. E.g. "mpirun -np 8 python MyScript.py" --> Generates traj.h5 file containing trajcetories, in the p4dir.

"""

import numpy as np
import h5py
import os
from datetime import datetime as dt
import time
import scipy.constants as sc
import traceback
import numpy.lib.recfunctions as rfn
try:
    from mpi4py import MPI
except:
    print("WARNING: MPI4PY FAILED TO LOAD. DO NOT CALL PARALLEL MPI FUNCTIONS.")

import lspsuite as ls
    
def hashdict(frame, dims=['xi','yi','zi','q']):
    """ Define the hashing dictionary (that specifies how to generate hashes). Call this only on the first frame.
    Inputs:
        frame : first pmovie frame
        dims  : list of dimensions you think may be relevant to hashing [if they aren't available or useful in this sim, they will be handled and removed]
    Outputs:
        hashd : dictionary of everything needed to generate hashes from the genhash function
    """
    
    # Check that these dimensions are relevant, and find the typical differences between values along axis
    meandiffs = np.zeros(1, dtype = frame['data'].dtype)
    dims2 = list(dims) # This will hold just the dims to be used in hashing
    for dim in dims:
        try:
            frame['data'][dim]
        except:
            print("Note: Dimension " + str(dim) + " not available in this pmovie frame. Removing from hash function.")
            dims2.remove(dim)
            continue
        
        meandiff = np.mean(np.diff(np.sort(frame['data'][dim])))
        if meandiff > 0:
            meandiffs[dim] = meandiff
        else: # meandiff is 0
            print("Note: Dimension " + str(dim) + " has no value in hashing based on this pmovie frame. Removing from hash function.")
            dims2.remove(dim) # This dimension has no value in hashing, so remove it
    
    # Redefine data and meandiffs, in case any new dimensions were removed above
    meandiffs = meandiffs[dims2]
    data = frame['data'][dims2]
    
    # Turn each dimension of data into integer form
    # data_int = (data - offsets) * scalers
    scalers = np.zeros(len(dims2), dtype=np.float)
    offsets = np.zeros(len(dims2), dtype=np.float)
    data_int = np.zeros((len(dims2), data.shape[0]), dtype=np.int64)
    pows = np.zeros(len(dims2), dtype=np.int)
    
    i = 0
    for dim in dims2:
        scalers[i] = 10.0 / meandiffs[dim] # The 10.0 is thrown in to help separate out similar values with differences up to 10x less than the average
        offsets[i] = np.min(data[dim], axis=0) # Offsets of data axes
        data_int[i] = ((data[dim] - offsets[i]) * scalers[i]).round().astype(np.int64) # Data in integer form
        i+=1
    
    # Give exponential multipliers to separate out the segments
    span = np.ceil(np.log10(np.max(data_int, axis=1))).astype(np.int64) # Orders of magnitude (powers of ten) needed to represent all the integers of this dimension
    cumpow = 0 # Cumulative power requirement
    for i in range(len(dims2)):
        pows[i] = cumpow + span[i]
        cumpow += span[i] + 1
    
    hashd = dict(dims=dims2, offsets=offsets, scalers=scalers, pows=pows)

    # Add a list of blacklisted duplicates to the hashd
    hashes = genhash(frame, hashd, removedupes=False)
    uni, counts = np.unique(hashes,return_counts=True);
    hashd.update({'dupes': uni[counts>1]})
    
    return hashd

def genhash(frame, hashd, removedupes=False):
    """ Generate all hashes for this frame
    Generate the hashes for the given frame for a specification
    given in the dictionary d returned from firsthash.    
    -----------
      frame :  frame to be hashed
      hashd :  hash specification generated from hashdict() on first frame
    Keywords:
    ---------
      removedupes: put -1 in duplicates

    Returns an array of the shape of the frames with hashes.
    """
    dims = hashd['dims']
    scalers = hashd['scalers']
    offsets = hashd['offsets']
    pows = hashd['pows']
    data = np.zeros((frame['data'].shape[0], len(dims)), dtype=np.float)
    
    for i in range(len(dims)):
        data[:,i] = frame['data'][dims[i]]

    hashes = ((data - offsets) * scalers * 10**pows).sum(axis=1)

    if removedupes:
        dups = np.in1d(hashes, hashd['dupes'])
        hashes[dups] = -1
    
    return hashes
    
def addhash(frame, hashd, removedupes=False):
    '''
    [Copied from Gregory Ngirmang's lspreader.
    https://github.com/noobermin/lspreader]
    helper function to add hashes to the given frame
    given in the dictionary d returned from firsthash.
    Parameters:
    -----------
      frame :  frame to hash.
      d     :  hash specification generated from firsthash.
    Keywords:
    ---------
      removedupes: put -1 in duplicates
    
    Returns frame with added hashes, although it will be modified in
    place
    '''
    hashes = genhash(frame, hashd, removedupes)
    frame['data'] = rfn.rec_append_fields(
        frame['data'],'hash',hashes)
    return frame

def sortone(fn, hashd=None):
    """ Loads one pmovie file and hashes it, then sorts it by hash. Duplicately-hashed particles are deleted from the output list, with prejudice.
    Assumes there is only one time step per pmovie file. Otherwise, takes only the first of these time steps.
    Inputs:
        fn: string, filename e.g.  fn="../../pmovie0004.p4(.gz)"
        hashd: The hashing dictionary greated by genhash(). If None, assume this is the first frame and defines future hashing.
    Outputs:
        data: 1D numpy array with several dtype fields, equivalent to the output of one frame of lspreader's read_pmovie(), but sorted by initial particle position
        stats: A dict containing information such as step number, step time, and number of particles
    """
    frame = ls.read_pmovie(fn)[0] # Read in the pmovie file using lspreader2. Assume first frame is the only frame.

    if hashd is None: # First pmovie file; define the hashing functions/parameters into a dict called "hashd"
        hashd = hashdict(frame)

    # Splice "hash" field as a frame['data'] field
    frame = addhash(frame, hashd, removedupes=True)

    # Sort by hash
    frame['data'].sort(order='hash')

    data = frame['data'][frame['data']['hash'] != -1]
    print("Fraction of duplicates:", (0.0 + len(frame['data']) - len(data))/len(frame['data']))

    del(frame['data']) # Remove the reference to the data array from the frame dictionary
    stats = frame # Refer to this array as the "stats" dictionary, which no longer contains the data

    return data, stats, hashd

def fillgaps(data, data_ref, shuff):
    """ Fill gaps (missing particles) in "data" with particles from "data_ref". Fill in missing particles in data.
    That is, those present in data_ref but not in data will be taken from data_ref and inserted into data.
    Also, shuffles the deck according to a random, but pre-agreed, permutation.
    Inputs:
        Note - inputs and outputs are all 1D numpy array with several dtype fields, equivalent to the output of one frame of lspreader's read_pmovie(), but sorted by initial particle position
        data: sorted data array with missing particles
        data_ref: sorted data array with all particles (the template, e.g. from previous time step) (equal to in length or longer than data)
        shuff: An array the same length of data_ref, with indices shuffled. Get via "shuff=np.random.permutation(len(data_ref))"
    Outputs:
        data_new: sorted data array which is a blend of data_ref and data, where missing particles have been replaced.
    """
    
    goodcdt = np.zeros(data_ref.shape, dtype=bool) # Allocate an array where present (not missing) particles will be marked as True. Initialize to False (all particles missing).
    ix1 = 0 # The index for data_ref
    ix2 = 0 # The index for data (will always be equal to or less than ix1)
    while ix2 < len(data):
        # Iterate over each of the reference particles, checking if they are missing from the data.
        # For each reference particle, check if any xi, zi, or yi are unequal
        if data[ix2]['hash'] != data_ref[ix1]['hash']:
            pass # This is a missing particle
        else: # This is not a missing particle. It (or its duplicate) have been there.
            goodcdt[ix1] = True # Mark as  a good trajectory
            ix2 += 1 # Move onto the next data particle only if the particle wasn't missing. Otherwise, stay on this data particle.
        ix1 += 1 # Every iteration, move on to the next reference particle

    # Copy into the new array, dat3s. Could also just do a rolling update.
    data_new = np.copy(data_ref) # Make a deep copy of the reference array, as your new output array
    data_new[goodcdt] = data # Fill in all particles that were present in your data (the missing particles will stay what they were in the reference data)

    print("Fraction missing:" + str(1 - float(len(goodcdt[goodcdt]))/float(len(goodcdt))))

    return data_new[shuff], goodcdt[shuff]  # Return the data array, with all particles now present (gaps are filled), and shuffled

def mpitraj(p4dir, h5fn = None, skip=1, start=0, stop=None):
    """ Create trajectories from a folder of pmovie files. Save into a "traj.h5" file.
    Assume we have greater than one processor. Rank 0 will do the hdf5 stuff"""
    
    # Set some basic MPI variables
    nprocs = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()
    comm = MPI.COMM_WORLD
    print("Number of processors detected through MPI:" + str(nprocs))
    
    # Check that we have at least two processors in our MPI setup    
    if nprocs < 2:
        raise Exception("Not enough processors. Need at least two, or call some other function.")
    
    # Everyone, get your bearings on the task to be performed
    # Get a sorted list of filenames for the pmovie files
    fns = ls.listp4(p4dir, prefix = "pmovie")[start:stop:skip] # Get list of pmovieXX.p4 files, and get the list sorted in ascending time step order
    nframes = len(fns) # Number of pmovie frames to prepare for

    # Rank0: Read in the first file and get reference data, spread that around  
    if rank == 0: # Rank 0, start working on that HDF5
        # Read the first frame, to get info like the length
        print("Rank 0 speaking, I'm reading in the first frame. Everyone else sit tight.")
        data_ref, _, hashd = sortone(fns[0])
        shuff = np.random.permutation(len(data_ref))
        print("Ok, I'm going to spread that around, now.")
    else:
        data_ref = None
        hashd = None # These "None" declarations are essential if we are to broadcast
        shuff = None
    data_ref = comm.bcast(data_ref, root=0)
    hashd = comm.bcast(hashd, root=0)
    shuff = comm.bcast(shuff, root=0)
    
    nparts = len(data_ref) # Number of particles extracted from this first frame (which will determine the rest, as well)

    # Assign files to each processor
    framechunks = ls.chunk(range(nframes), nchunks=(nprocs - 1), shuffle=True)
    myframes = np.array(framechunks[rank - 1], dtype='i') #TODO: Better ordering of chunks (non-sequential) # If there are 4 processors, break into 3 chunks (as 0th processor just writes files)
    
    if rank == 0: # Rank 0, start working on that HDF5
        print("Good stuff. I'm going to get started on this HDF5; I'll let you know how it goes.")
        # Determine the filename for output of hdf5 trajectories file
        if not h5fn:    
            h5fn = os.path.join(p4dir, 'traj.h5')

        t1 = dt.now()
        #goodkeys = ['xi', 'zi', 'x', 'z', 'ux', 'uy', 'uz','q'] # 2D X-Z 3v
        #goodkeys = ['xi', 'x', 'ux','q'] # 1D X only
        goodkeys_tmp = ['xi', 'yi', 'zi', 'x', 'y', 'z', 'ux', 'uy', 'uz', 'q']
        goodkeys = list(goodkeys_tmp) # Copy
        print("Data ref dtype: " + str(data_ref.dtype))
        for key in goodkeys_tmp: # Remove any keys that are not usable in this simulation.
            try:
                data_ref[key]
            except:
                print("Dimension " + str(key) + " is not available in this simulation... skipping.")
                goodkeys.remove(key)
            
        #goodkeys = dims + idims + ['ux', 'uy', 'uz', 'q']
        # Open the HDF5 file, and step over the p4 files
        with h5py.File(h5fn, "w") as f:

            # Allocate the HDF5 datasets
            f.create_dataset("t", (nframes,), dtype='f')
            f.create_dataset("step", (nframes,), dtype='int32')
            f.create_dataset("gone", (nframes, nparts,), dtype='bool', chunks=True) # Chunking makes later retrieval along _both_ dimensions reasonably quick
            for k in goodkeys:
                f.create_dataset(k, (nframes, nparts,), dtype='f', chunks=True)
        
            # Now, iterate over the files (collect their data and save to the HDF5)
            for i in range(nframes):
                print("Rank 0: Collecting file " + str(i) + " of " + str(nframes))  
                t2 = dt.now()
                datdict = comm.recv(source=MPI.ANY_SOURCE, tag=i)  # Retrieve the result
                t3 = dt.now()
                datnew = datdict['datnew']
                stats = datdict['stats']
                goodcdt = datdict['goodcdt']
                badcdt = np.logical_not(goodcdt) # Flip the sign of good condit
                #datnew[badcdt] = data_ref[badcdt] # Fill in the missing particles
                t4 = dt.now()

                f['t'][i] = stats['t']
                f['step'][i] = stats['step']
                f['gone'][i] = badcdt # Flag the particles that were missing
                for k in goodkeys:
                    f[k][i] = datnew[k]
                        
                t5 = dt.now()
                print("Seconds on receipt, analysis, storage: " + str((t3 - t2).total_seconds()) + str(" ") + str((t4 - t3).total_seconds()) + " " + str((t5 - t4).total_seconds()))
                data_ref = datnew # The new array becomes the reference for next iteration
            print("ELAPSED TIME (secs) " + str((dt.now() - t1).total_seconds()))
            print("HDF5 file filled with particle trajectories stored at: " + h5fn)
    else: # All other processes (non-rank 0) do the opening and reading of pmovie files, passing this info back
        print("I am a servant of the ranks.")
        print("Rank " + str(rank) + str(": I have frames:") + str(myframes))
        for i in range(len(myframes)):
            ix = myframes[i] # The index that refers to the entire list (of all files)
            print("Rank " + str(rank) + str(": working on file ") + str(ix))
            fn = fns[ix]
            dattmp, stats, _ = sortone(fn, hashd=hashd) # Read (and sort) pmovie file
            datnew, goodcdt = fillgaps(dattmp, data_ref, shuff) # fill in missing particles (according to data_ref)
            datdict = {}
            datdict['datnew'] = datnew
            datdict['stats'] = stats
            datdict['goodcdt'] = goodcdt
            comm.send(datdict, dest=0, tag=ix) # Note: comm.isend would give an EOFError, for some reason, so don't use it.
