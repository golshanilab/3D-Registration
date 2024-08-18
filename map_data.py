import tifffile as tif
import numpy as np
import zarr
import multiprocessing
import os
import sys

def map_data(imageData, start_idx, files, fname):

    for i, file in enumerate(files):
        image = tif.imread(os.path.join(fname,file))

        imageData[start_idx+i,:, :,:] = image

if __name__ == '__main__':

    fname = os.path.normpath(str(sys.argv[1]))

    files = [f for f in os.listdir(fname) if f.endswith('.tif')]
    files.sort(key=lambda x: '{0:0>20}'.format(x).lower())
    length = len(files)
    cores = multiprocessing.cpu_count()-4

    files_per_core = int(length//cores)
    frame = tif.imread(os.path.join(fname,files[0]))
    n,m,p = frame.shape
    data_type = frame.dtype

    mapped_file_path = os.path.join(os.path.dirname(fname), 'mapped_data.zarr')
    synch_file_path = os.path.join(os.path.dirname(fname), 'mapped_data.sync')

    sync = zarr.ProcessSynchronizer(synch_file_path)
    imageData = zarr.open_array(mapped_file_path, mode='a', shape=(length,n,m,p), chunks=(1,n,m,p), dtype=data_type, synchronizer=sync)

    procs = np.empty(cores, dtype=object)

    for i in range(cores):
        start_idx = i*files_per_core
        if i == cores-1:
            end_idx = length
        else:
            end_idx = (i+1)*files_per_core

        procs[i] = multiprocessing.Process(target=map_data, args=(imageData, start_idx, files[start_idx:end_idx], fname))
        procs[i].start()

    for i in range(cores):
        procs[i].join()
        
