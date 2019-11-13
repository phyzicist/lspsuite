""" Chunking and unchunking lists is helpful when splitting lists up for parallel processing.

Written largely by users of StackOverFlow questions; copied here by Scott Feister

Adapted by Scott Feister 11-12-2019

"""

def chunk(mylist, nchunks=None, chunksize=None, shuffle=False):
    """
    Break a list into nearly-equal-length sublists. Some sublists may be empty.
    
    Will return a list of lists that can be flattened back into a single list by "unchunk()".
    
    Either nchunks OR chunksize must be specified, but NOT both.
       
    The input "shuffle" only applies if chunksize is specified.
    
    Note that is shuffle is True, unchunk() will not be a perfect undo of chunk().
    
    Example Usage:
        mylist = [0,1,2,3,4,5,6]
        chunk(mylist, nchunks=4, shuffle=False)  => [[0, 1], [2, 3], [4, 5], [6]]
        chunk(mylist, nchunks=4, shuffle=True)  => [[0, 4], [1, 5], [2, 6], [3]]
        chunk(mylist, chunksize=3)  => [[0, 1, 2], [3, 4, 5], [6]]
    
    """
    
    if nchunks is not None and chunksize is not None:
        raise Exception("Specify only nchunks OR chunksize, but NOT both.")
    
    if nchunks is not None and shuffle:
        # Divide mylist into nchunks sublists, shuffled in ordering. """
        chunks = [None]*nchunks
        for i in range(nchunks):
            chunks[i] = []
            
        i = 0
        while i < len(mylist):
            j = 0
            while (i < len(mylist)) & (j < nchunks):
                chunks[j].append(mylist[i])
                j += 1
                i += 1
        return chunks
    
    elif nchunks is not None:
        # Divide mylist into nchunks sublists, sequentially. """
        # Modified from http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
        avg = len(mylist) / float(nchunks)
        chunks = []
        last = 0.0

        while last < len(mylist):
            chunks.append(mylist[int(last):int(last + avg)])
            last += avg

        return chunks
        
    elif chunksize is not None:
        # Break mylist into fixed, chunksize-sized chunks. The final element of the new list will be chunksize-sized or less
        # Modified from http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
        chunks = []
        for i in range(0, len(mylist), chunksize):
            chunks.append(mylist[i:i+chunksize])
        return chunks

def unchunk(chunks):
    """Flatten the first dimension of a list. E.g. if input is l = [[1,2,],[3,4]], output is [1,2,3,4]"""
    # Copied from http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
    return [item for sublist in chunks for item in sublist]
    
