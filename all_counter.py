import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.feature import peak_local_max

from sklearn import metrics
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
        # row = joint.shape[0]
        # col = joint.shape[1]
        #
        # window = int(sys.argv[1])
        # if window == 1:
        #     for r in range(row):
        #         for c in range(window,col):
        #             if joint[r][c] > joint[r][c-window]:
        #                 joint[r][c-window] = 0
        #             elif joint[r][c] < joint[r][c-window]:
        #                 joint[r][c] = 0
        #
        #     for c in range(col):
        #         for r in range(window,row):
        #             if joint[r][c] > joint[r-window][c]:
        #                 joint[r-window][c] = 0
        #             elif joint[r][c] < joint[r-window][c]:
        #                 joint[r][c] = 0
        # else:
        #     for r in range(row):
        #         for c in range(window,col):
        #             max_index = np.argsort(joint[r][c-window:c])[-1]
        #             for ind in range(c-window,c):
        #                 if ind != c-window + max_index:
        #                     joint[r][ind] = 0
        #
        #     joint = np.transpose(joint)
        #
        #     for r in range(joint.shape[0]):
        #         for c in range(window,joint.shape[1]):
        #             max_index = np.argsort(joint[r][c-window:c])[-1]
        #             for ind in range(c-window,c):
        #                 if ind == c - window + max_index:
        #                     continue
        #                 else:
        #                     joint[r][ind] = 0
        #     joint = np.transpose(joint)
        all = np.maximum(joint,all)


    # all = np.nonzero( all > 0.3)
    points = peak_local_max(all,threshold_rel = 0.5)
    if len(points) != 0:
        # points = np.array([all[0][0],all[1][0]])
        #
        # for i in range(1,all[0].shape[0]):
        #     if i == 1:
        #         points = np.append([points],[[all[0][i],all[1][i]]], axis=0)
        #     else:
        #         points = np.append(points,[[all[0][i],all[1][i]]], axis=0)
        #
        # if len(all[0]) == 1:
        #     points = points.reshape(1,-1)
        #

        # af = AffinityPropagation(preference = -2800).fit(points)
        # cluster_centers_indices = af.cluster_centers_indices_
        # labels = af.labels_

        n_clusters_ = len(points)

    else:
        n_clusters_ = 0

        if not (n_clusters_ in counter):
            counter[n_clusters_] = 0
        counter[n_clusters_] += 1

    return ([n_clusters_,int(re.search(r'p\d+',smap).group()[1:]) +1])


files = os.listdir("./MC "+ sys.argv[2])
# for i in files:
#     if i.startswith("smap"):


results = parallel_process(files, fish_count)

results = sorted(results, key=takeFirst)

with open('count_maxi0.txt', 'w') as f:
    for item in results:
        f.write("{},{}\n".format(item[1],item[0]))
