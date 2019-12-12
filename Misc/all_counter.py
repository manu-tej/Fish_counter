import numpy as np
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist

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
    scores = np.load("./MC "+sys.argv[1]+"/"+smap)
    global counter

    num_of_joints = scores.shape[2]

    k=[]

    all = np.zeros(scores[:,:,0].shape)

    for j in range(0,1):
        k.append(scores[:,:,j])

    for joint in k:
        all = np.maximum(joint,all)

    points = peak_local_max(all,threshold_rel = 0.9)
    if len(points) != 0:

        n_clusters_ = len(points)

    else:
        n_clusters_ = 0

        if not (n_clusters_ in counter):
            counter[n_clusters_] = 0
        counter[n_clusters_] += 1

    return ([points,int(re.search(r'p\d+',"./MC "+sys.argv[1]+"/"+smap).group()[1:]) +1])


files = os.listdir("./MC "+ sys.argv[1])

results = parallel_process(files, fish_count)
results = sorted(results, key=takeFirst)

results = np.array(results)
coor,frames = results.T

cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors.Normalize(vmin=min(frames), vmax=max(frames))
colors = [cmap(normalize(value)) for value in frames]
colors.reverse()

fig, ax = plt.subplots(figsize=(10,10))

ax.set_ylim(122,0)
ax.set_xlim(0,162)
plt.title('Path traced by noses in a typical chase behavior')

count = 0
maximum_fish = 0
for i in coor:
    count += 1
    if count == 1:
        temp = i
        print(temp)
        no_of_markers = len(i)
        maximum_fish = no_of_markers
        cl = colors.pop()
        for fish in range(no_of_markers):
            x,y = i[fish]
            ax.scatter(y,x,color=cl,marker = matplotlib.markers.MarkerStyle.filled_markers[fish], s =15)
    else:
        print(count)
        second_list = np.array(i)
        max_fish = len(second_list)
        cl = colors.pop()
        temp_1 = []
        tracker = 0
        while (len(temp) != 0 and max_fish > 0):
            distance = cdist([temp[0]], second_list)
            print(temp)
            print(temp[0])
            print(second_list)
            print(distance)
            rem_idx = np.unravel_index(np.argmin(distance),distance.shape)
            x,y = second_list[rem_idx[1]]
            ax.scatter(y,x,color=cl,marker = matplotlib.markers.MarkerStyle.filled_markers[rem_idx[0]+tracker], s =15)
            tracker += 1
            temp_1.append([second_list[rem_idx[1]][0],second_list[rem_idx[1]][1]])
            temp = np.delete(temp,rem_idx[0],0)
            max_fish -= 1
            print(temp_1)
        temp = temp_1
        if maximum_fish < len(temp):
            maximum_fish = len(temp)

cax, _ = matplotlib.colorbar.make_axes(ax)
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
cbar.ax.set_ylabel('Time in frames', rotation=270)
ax.set_ylabel('Height (pixels)')
ax.set_xlabel('Width (pixels)')
legend_markers = []
all_fish = maximum_fish
while(maximum_fish != 0):
    legend_markers.append(mlines.Line2D([], [], color='black', marker=matplotlib.markers.MarkerStyle.filled_markers[all_fish - maximum_fish], linestyle='None',
                          markersize=5, label='Fish '+ str(all_fish - maximum_fish + 1)))
    maximum_fish -= 1

ax.legend(handles=legend_markers)

plt.show()
