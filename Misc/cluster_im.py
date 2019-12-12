import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import umap
import seaborn as sns
import csv


sns.set(style='white', context='poster')

with open('labels.csv', mode='r') as infile:
    reader = csv.reader(infile, delimiter=',')
    with open('test.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        mydict = {rows[1]:rows[0] for rows in reader}
        vid_list = [rows[1] for rows in reader]

with open('labels.csv', mode='r') as infile:
    reader = csv.reader(infile, delimiter=',')
    with open('test.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        vid_list = [rows[1] for rows in reader]

vid_list = os.listdir('plot-poses')
print(vid_list[0])
img = mpimg.imread(os.path.join('plot-poses',vid_list[0],'trajectory.png'))
img.resize(400,600,4)
target = []
target.append(1)

colori = {'b':'blue', 'c':'green', 'f':'black', 'm':'orange', 'o':'cyan', 'p':'red', 'r':'yellow'}
conver = {'b':1, 'c':0, 'f':3, 'm':4, 'o':5, 'p':6, 'r':7}

classes = ['build multiple', 'scoop', 'feed spit', 'feed multiple', 'other', 'build spit', 'spit-run']
for i in range(1,200):
    print(vid_list[i])
    temp = mpimg.imread(os.path.join('plot-poses',vid_list[i],'trajectory.png'))
    temp.resize(400,600,4)
    img = np.concatenate((img, temp), axis = 0)
    target.append(conver[mydict[vid_list[i]]])


img = img.reshape(200,400,600,4)

print(img.shape)
k = img.reshape(200,400*600*4)

embedding = umap.UMAP(n_neighbors=35).fit_transform(data)

embedding = umap.UMAP(n_neighbors=35, random_state =123, target_metric = 'categorical', verbose = True).fit_transform(data, y=target)
fig, ax = plt.subplots(1, figsize=(14, 10))
plt.scatter(*embedding.T, s= 10, c=target, cmap='Spectral', alpha=0.85)
plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(8)-0.5)
cbar.set_ticks(np.arange(7))
cbar.set_ticklabels(classes)
plt.title('Path trajectories as 2D points')
plt.show()

a = np.genfromtxt("csv/214_1648_175_136DeepCut_resnet50_fishv3Aug3shuffle1_1030000.csv",delimiter=',', skip_header=3)
a = a[:,1:]

conver = {'b':0, 'c':1, 'f':2, 'm':3, 'o':4, 'p':5, 'r':6}

print(vid_list[0])
data = np.genfromtxt(os.path.join('csv',vid_list[0] + 'DeepCut_resnet50_fishv3Aug3shuffle1_1030000.csv'), delimiter=',', skip_header=3)
data = data[:,1:]
data = data.reshape(480,-1)
data[np.all(data > 0.7, axis = 1)] = 0
data = data[:,:2]
target = []
target.append(conver[mydict[vid_list[0]]])



classes = ['build multiple', 'scoop', 'feed spit', 'feed multiple', 'other', 'build spit', 'spit-run']
for i in range(1,2000):
    print(vid_list[i])
    temp = np.genfromtxt(os.path.join('csv',vid_list[i] + 'DeepCut_resnet50_fishv3Aug3shuffle1_1030000.csv'), delimiter=',', skip_header=3)
    temp = temp[:,1:]
    temp = temp.reshape(480,-1)
    temp[np.all(temp > 0.7, axis = 1)] = 0
    temp = temp[:,:2]
    data = np.concatenate((data, temp), axis = 0)
    target.append(conver[mydict[vid_list[i]]])

data = data.reshape(2000,-1)
