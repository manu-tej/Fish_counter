#!/usr/bin/env python3

import numpy as np
import sys
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.feature import peak_local_max

import re
import time

def parallel_process(array, function, n_jobs=4, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar.

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

def takeFirst(elem):
    return elem[1]

def fish_count(smap) :
    scores = np.load(smap)
    global counter

    num_of_joints = scores.shape[2]

    k=[]

    all = np.zeros(scores[:,:,0].shape)

    for j in range(0,1):
        k.append(scores[:,:,j])

    for joint in k:
        all = np.maximum(joint,all)


    # all = np.nonzero( all > 0.3)
    points = peak_local_max(all,threshold_rel = 0.5)
    if len(points) != 0:
        n_clusters_ = len(points)

    else:
        n_clusters_ = 0

        if not (n_clusters_ in counter):
            counter[n_clusters_] = 0
        counter[n_clusters_] += 1

    return ([n_clusters_,int(re.search(r'p\d+',smap).group()[1:]) +1])


files = os.listdir("./MC "+ sys.argv[2])



results = parallel_process(files, fish_count)

results = sorted(results, key=takeFirst)

with open('count_maxi0.txt', 'w') as f:
    for item in results:
        f.write("{},{}\n".format(item[1],item[0]))
