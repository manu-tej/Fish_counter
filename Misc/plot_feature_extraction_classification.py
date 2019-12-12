"""
UMAP as a Feature Extraction Technique for Classification
---------------------------------------------------------

The following script shows how UMAP can be used as a feature extraction
technique to improve the accuracy on a classification task. It also shows
how UMAP can be integrated in standard scikit-learn pipelines.

The first step is to create a dataset for a classification task, which is
performed with the function ``sklearn.datasets.make_classification``. The
dataset is then split into a training set and a test set using the
``sklearn.model_selection.train_test_split`` function.

Second, a linear SVM is fitted on the training set. To choose the best
hyperparameters automatically, a gridsearch is performed on the training set.
The performance of the model is then evaluated on the test set with the
accuracy metric.

 Third, the previous step is repeated with a slight modification: UMAP is
 used as a feature extraction technique. This small change results in a
 substantial improvement compared to the model where raw data is used.
"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm
from umap import UMAP
import csv
import numpy as np
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import neighbors
from sklearn import ensemble



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


# Make a toy dataset

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=42)

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

x_train = UMAP(n_neighbors=35, n_components= 400,random_state = 456).fit_transform(X_train, y=y_train)
mapper = UMAP(n_neighbors=35,n_components= 400, random_state = 456).fit(X_train, y=y_train)
x_test = mapper.transform(X_test)



# Classification with a linear SVM


clf = neighbors.KNeighborsClassifier(29, weights='distance', metric='manhattan')
clf.fit(X_train, y_train)
print("Accuracy on the test set with raw data: {:.3f}".format(
    clf.score(X_test, y_test)))

svc = neighbors.KNeighborsClassifier()
params_grid = {"n_estimators": [k for k in range(10,100,10)],
                "max_depth": [k for k in range(1,7)]}
clf = GridSearchCV(clf, params_grid, verbose=1, cv=10, n_jobs=-1)
results = clf.fit(x_train, y_train)
print("Accuracy on the test set with raw data: {:.3f}".format(
    clf.score(X_test, y_test)))

svc = svm.SVC(kernel='linear')
params_grid = {"C": [10**k for k in range(-3, 4)]}
clf = GridSearchCV(svc, params_grid)
clf.fit(x_train, y_train)
print("Accuracy on the test set with raw data: {:.3f}".format(
    clf.score(x_test, y_test)))

classif = OneVsRestClassifier(clf, n_jobs=-1)
classif.fit(x_train, y_train)
print("Accuracy on the test set with raw data: {:.3f}".format(
    classif.score(x_test, y_test)))

# Transformation with UMAP followed by classification with a linear SVM
umap = UMAP(random_state=456, )
pipeline = Pipeline([("umap", umap),
                     ("svc", svc)])
params_grid_pipeline = {"umap__n_neighbors": [25, 35],
                        "umap__n_components": [15, 25, 50],
                        "svc__C": [10**k for k in range(-3, 4)]}

clf_pipeline = GridSearchCV(pipeline, params_grid_pipeline)
clf_pipeline.fit(X_train, y_train)
print("Accuracy on the test set with UMAP transformation: {:.3f}".format(
    clf_pipeline.score(X_test, y_test)))
