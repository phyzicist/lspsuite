# Functions relating to LSP file paths, making directories, and filename manipulations

import os
import re
import numpy as np

def subdir(folder, name):
    """ Make a subdirectory in the specified folder, if it doesn't already exist"""
    subpath = os.path.join(folder,name)
    if not os.path.exists(subpath):
        try:
            os.mkdir(subpath)
        except:
            if not os.path.exists(subpath):
                raise
    return subpath

def listp4(folder, prefix = 'flds'):
    """ Get a sorted list of full path filenames for all files 'fldsXXX.p4(.gz)' (for prefix = 'flds') in a folder, sorted by number XXX
    
    Inputs:
       folder: a file path to the folder, e.g. "C:/run1", which contains P4 outputs from LSP simulations
       prefix: string with which the filename starts, e.g. 'pmovie' to return files matching '.../pmovie*.p4'
    Outputs:
       fns: a list of (sorted) full paths to files matching 'fldsXXX.p4', e.g. fns = ['C:/run1/flds1.p4', 'C:/run1/flds5.p4', 'C:/run1/flds10.p4']   
    """
    
    fns = [] # Will store the list of filenames
    nums = [] # Will store a list of the XXX numbers in 'fldsXXX.p4'
    
    pattern = r'^(' + prefix + ')(\d*?)(\.p4)(\.gz){0,1}$'
    # If prefix = 'flds', matches 'fldsXXX.p4' and 'fldsXXX.p4.gz', where XXX is a number of any length. Does not match 'dfldsXXX.p4' or 'fldsXXX.p4.zip'
    

    for name in os.listdir(folder): # note 'file' can be a file, directory, etc.
        m = re.search(pattern, name)
        if m: # If the filename matches the pattern, add this file to the list
            fns.append(os.path.join(folder, name))
            
            # Extract "XXX" as a number from "fldsXXX.p4(.gz)"
            # Note: re.split(r'^(fld)(\d*?)(\.p4)(\.gz){0,1}$', 'flds5020.p4.gz) returns something like ['', 'fld', '5020', '.p4', '.gz', ''], so we take the element at index 2 and convert to integer
            fnum = int(re.split(pattern, name)[2])
            nums.append(fnum)

    # Sort the field names by their XXX number    
    idx = np.argsort(nums) # Get a list of sorted indices such that the filenames will be sorted correctly by time step
    fns = np.array(fns)[idx].tolist() # Convert filenames list to a numpy string array for advanced indexing, then convert back to a python list
    return fns

def listp4pairs(folder):
    """ Get a 1-to-1 list of field and scalar filenames in a directory, and exclude any fld or scl files that are missing their partner.
    
    If fld000.p4 exists but scl000.p4 does not, this will exclude fld000.p4 from the list of "fns_fld".
    """
    fld = listp4(folder, 'flds')
    scl = listp4(folder, 'sclr')
    
    fns_fld = []
    fns_scl = []

    for fn_scl in scl:
        folder, name = os.path.split(fn_scl)
        name_fld = name.replace('sclr','flds')
        fn_fld = os.path.join(folder, name_fld)
        if fn_fld in fld:
            fns_fld.append(fn_fld)
            fns_scl.append(fn_scl)
    
    return fns_fld, fns_scl