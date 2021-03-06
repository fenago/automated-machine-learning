
Lab 4: Automated Algorithm Selection
====================================


#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

All notebooks are present in `lab 04` folder.


Computational complexity
========================

Computational efficiency and complexity are important aspects of
choosing ML algorithms, since they will dictate the resources needed for
model training and scoring in terms of time and memory requirements.

For example, a compute-intensive algorithm will require a longer time to
train and optimize its hyperparameters. You will usually distribute the
workload among available CPUs or GPUs to reduce the amount of time spent
to acceptable levels.

In this section, some algorithms will be examined in terms of these
constraints but, before getting into deeper details of ML algorithms,
you need to know the basics of the complexity of an algorithm.


The complexity of an algorithm will be based on its input size. For ML
algorithms, this could be the number of elements and features. You will
usually count the number of operations needed to complete the task in
the worst-case scenario and that will be your algorithm\'s complexity.




Big O notation
==============

You have probably heard of big O notation. It has different classes for
indicating complexity such as linear---[O(n)],
logarithmic---[O(log n)], quadratic---[O(n2)],
cubic---[O(n3)], and similar classes. The reason you use big O is
because the runtime of algorithms is highly dependent on the hardware
and you need a systematic way of measuring the performance of an
algorithm based on the size of its input. Big O looks at the steps of an
algorithm and figures out the worst-case scenario as mentioned.

For example, if [n] is the number of elements that you would like
to append to a list, its complexity is [O(n)], because the number
of appended operations depends on the [n]. The following code
block will help you to plot how different complexities grow as a
function of their input size:


```
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the style of the plot
plt.style.use('seaborn-whitegrid')

# Creating an array of input sizes
n = 10
x = np.arange(1, n)

# Creating a pandas data frame for popular complexity classes
df = pd.DataFrame({'x': x,
                   'O(1)': 0,
                   'O(n)': x,
                   'O(log_n)': np.log(x),
                   'O(n_log_n)': n * np.log(x),
                   'O(n2)': np.power(x, 2), # Quadratic
                   'O(n3)': np.power(x, 3)}) # Cubic

# Creating labels
labels = ['$O(1) - Constant$',
          '$O(\log{}n) - Logarithmic$',
          '$O(n) - Linear$',
          '$O(n^2) - Quadratic$',
          '$O(n^3) - Cubic$',
          '$O(n\log{}n) - N log n$']

# Plotting every column in dataframe except 'x'
for i, col in enumerate(df.columns.drop('x')):
    print(labels[i], col)
    plt.plot(df[col], label=labels[i])

# Adding a legend
plt.legend()

# Limiting the y-axis
plt.ylim(0,50)

plt.show()
```


We get the following plot as the output of the preceding code:


![](./images/4604f644-a706-4b19-8226-c6cdf65c08ef.png)



One thing to note here is that there are some crossover points between
different levels of complexities. This shows the role of data size.
It\'s easy to understand the complexity of simple examples, but what
about the complexity of ML algorithms? If the introduction so far has
already piqued your interest, continue reading the next section.









Simple measure of training and scoring time
===========================================

Let's see a quick example of the **k-nearest neighbor** (**k-NN**)
algorithm, which works both for classification and regression problems.
When an algorithm scores a new feature vector, it checks the *k* nearest
neighbors and outputs a result. If it\'s a classification problem, a
prediction is made using a majority vote; if it\'s a regression problem,
then the average of the values is used as a prediction.

Let\'s understand this better by working on an example classification
problem. First, you will create a sample dataset and you will examine
the k-NN algorithm in terms of time spent for training and scoring.

Just to make things easier, the following function will be used to
measure the time spent on a given line:


```
from contextlib import contextmanager
from time import time

@contextmanager
def timer():
    s = time()
    yield
    e = time() - s
    print("{0}: {1} ms".format('Elapsed time', e))
```


You can use this function in the following way:


```
import numpy as np

with timer():
    X = np.random.rand(1000)
```


It outputs the time spent executing that line:


```
Elapsed time: 0.0001399517059326172 ms
```


Now, you can work with [KNeighborsClassifier] of the scikit-learn
library and measure the time spent for training and scoring:


```
from sklearn.neighbors import KNeighborsClassifier

# Defining properties of dataset
nT = 100000000 # Total number of values in our dataset
nF = 10 # Number of features
nE = int(nT / nF) # Number of examples

# Creating n x m matrix where n=100 and m=10
X = np.random.rand(nT).reshape(nE, nF)

# This will be a binary classification with labels 0 and 1
y = np.random.randint(2, size=nE)

# Data that we are going to score
scoring_data = np.random.rand(nF).reshape(1,-1)

# Create KNN classifier
knn = KNeighborsClassifier(11, algorithm='brute')

# Measure training time
with timer():
    knn.fit(X, y)

# Measure scoring time
with timer():
    knn.predict(scoring_data)
```


Let\'s see the output:


```
Elapsed time: 1.0800271034240723 ms
Elapsed time: 0.43231201171875 ms
```


Just to have an idea about how this compares to other algorithms, you
can try one more classifier, such as logistic regression:


```
from sklearn.linear_model import LogisticRegression
log_res = LogisticRegression(C=1e5)

with timer():
    log_res.fit(X, y)

with timer():
    prediction = log_res.predict(scoring_data)
```


The output for logistic regression is as follows:


```
Elapsed time: 12.989803075790405 ms
Elapsed time: 0.00024318695068359375 ms
```


It looks quite different! Logistic regression is slower in training and
much faster in scoring. Why is that?

You will learn the answer to that question but, before getting into the
details of the preceding results, let\'s talk a little about code
profiling in Python.



Code profiling in Python
========================

Some applications will require your machine learning models to be
performant in terms of training and scoring time. For example, a
recommender engine might require you to generate recommendations in less
than a second and if you have more than a second latency, profiling is
one way to understand intensive operations. Code profiling will help you
a lot to understand how different parts of your program are executed.
Profiling stats will give metrics, such as the number of calls, the
total time spent to execute a function call including/excluding calls to
its sub-functions, and incremental and total memory usage.

The [cProfile] module in Python will help you to see time
statistics for every function. Here\'s a small example:


```
# cProfile
import cProfile

cProfile.run('np.std(np.random.rand(1000000))')
```


In the previous line, standard deviation is calculated for 1,000,000
random samples that are drawn from uniform distribution. The output will
show time statistics for all the function calls to execute a given line:


```
23 function calls in 0.025 seconds
   Ordered by: standard name
   ncalls tottime percall cumtime percall filename:lineno(function)
        1 0.001 0.001 0.025 0.025 <string>:1(<module>)
        1 0.000 0.000 0.007 0.007 _methods.py:133(_std)
        1 0.000 0.000 0.000 0.000 _methods.py:43(_count_reduce_items)
        1 0.006 0.006 0.007 0.007 _methods.py:86(_var)
        1 0.001 0.001 0.008 0.008 fromnumeric.py:2912(std)
        2 0.000 0.000 0.000 0.000 numeric.py:534(asanyarray)
        1 0.000 0.000 0.025 0.025 {built-in method builtins.exec}
        2 0.000 0.000 0.000 0.000 {built-in method builtins.hasattr}
        4 0.000 0.000 0.000 0.000 {built-in method builtins.isinstance}
        2 0.000 0.000 0.000 0.000 {built-in method builtins.issubclass}
        1 0.000 0.000 0.000 0.000 {built-in method builtins.max}
        2 0.000 0.000 0.000 0.000 {built-in method numpy.core.multiarray.array}
        1 0.000 0.000 0.000 0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1 0.017 0.017 0.017 0.017 {method 'rand' of 'mtrand.RandomState' objects}
        2 0.001 0.001 0.001 0.001 {method 'reduce' of 'numpy.ufunc' objects}
```


[23] function calls are executed in [0.025] seconds and most
of the time is spent generating random numbers and calculating the
standard deviation, as you would expect.



Linearity versus non-linearity
==============================

Another consideration is decision boundaries. Some algorithms, such as
logistic regression or **Support Vector Machine** (**SVM**), can learn
linear decision boundaries while others, such as tree-based algorithms,
can learn non-linear decision boundaries. While linear decision
boundaries are relatively easy to calculate and interpret, you should be
aware of errors that linear algorithms will generate in the presence of
non-linear relationships.



Drawing decision boundaries
===========================

The following code snippet will allow you to examine the decision
boundaries of different types of algorithms:


```
import matplotlib.cm as cm

# This function will scale training datatset and train given classifier.
# Based on predictions it will draw decision boundaries.

def draw_decision_boundary(clf, X, y, h = .01, figsize=(9,9), boundary_cmap=cm.winter, points_cmap=cm.cool):

    # After you apply StandardScaler, feature means will be removed and all features will have unit variance.
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)

    # Splitting dataset to train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

    # Training given estimator on training dataset by invoking fit function.
    clf.fit(X_train, y_train)

    # Each estimator has a score function.
    # Score will show you estimator's performance by showing metric suitable to given classifier.
    # For example, for linear regression, it will output coefficient of determination R^2 of the prediction.
    # For logistic regression, it will output mean accuracy.

    score = clf.score(X_test, y_test)
    print("Score: %0.3f" % score)

    # Predict function of an estimator, will predict using trained model
    pred = clf.predict(X_test)

    # Figure is a high-level container that contains plot elements
    figure = plt.figure(figsize=figsize)

    # In current figure, subplot will create Axes based on given arguments (nrows, ncols, index)
    ax = plt.subplot(1, 1, 1)

    # Calculating min/max of axes
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Meshgrid is usually used to evaluate function on grid.
    # It will allow you to create points to represent the space you operate
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Generate predictions for all the point-pair created by meshgrid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # This will draw boundary
    ax.contourf(xx, yy, Z, cmap=boundary_cmap)

    # Plotting training data
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=points_cmap, edgecolors='k')

    # Potting testing data
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=points_cmap, alpha=0.6, edgecolors='k')

    # Showing your masterpiece
    figure.show()
```




Decision boundary of logistic regression
========================================

You can start with logistic regression to test this function:


```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# sklearn.linear_model includes regression models where target variable is a linear combination of input variables
from sklearn.linear_model import LogisticRegression

# make_moons is another useful function to generate sample data
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)

# Plot sample data
plt.scatter(X[:,0], X[:,1], c=y, cmap=cm.cool)
plt.show()
```


We get the following plot:


![](./images/633a0332-76b4-4a83-9bdd-6ce44db4267f.png)


Now, you can use the [draw\_decision\_boundary] function to
visualize the decision boundary for [LogisticRegression]:


```
draw_decision_boundary(LogisticRegression(), X, y)
```


It will output the following plot:


![](./images/7dbca387-4be3-495d-9610-5195af978b4f.png)


Logistic regression is a member of the generalized linear models and it
produces a linear decision boundary. A linear decision boundary is not
able to separate classes for such datasets. Logistic regression\'s
output is calculated based on the weighted sum of its inputs. Since
output doesn\'t depend on the product or quotient of its parameters, it
will produce a linear decision boundary. There are ways to overcome this
problem, such as regularization and feature mapping, but you can use
other algorithms in such cases that are able to work with non-linear
data.



The decision boundary of random forest
======================================

Random forest is a meta estimator, that will build many different models
and aggregate their predictions to come up with a final prediction.
Random forest is able to produce non-linear decision boundaries, since
there\'s no linear relationship between inputs and outputs. It has many
hyperparameters to play with but for the sake of simplicity, you will
use the default configuration:


```
from sklearn.ensemble import RandomForestClassifier

draw_decision_boundary(RandomForestClassifier(), X, y)
```


We get the following plot from the preceding code:


![](./images/f5a7d1f2-9b00-43d6-9a9a-ea89cd1ac805.png)


Not looking too bad at all! Every algorithm will provide you with
different decision boundaries, based on their inner workings, and you
should definitely experiment with different estimators to better
understand their behavior.



Unsupervised AutoML
===================

When your dataset doesn\'t have a target variable, you can use
clustering algorithms to explore it, based on different characteristics.
These algorithms group examples together, so that each group will have
examples as similar as possible to each other, but dissimilar to
examples in other groups.

Since you mostly don\'t have labels when you are performing such
analysis, there is a performance metric that you can use to examine the
quality of the resulting separation found by the algorithm.

It is called the **Silhouette Coefficient**. The Silhouette Coefficient
will help you to understand two things:

-   **Cohesion**: Similarity within clusters
-   **Separation**: Dissimilarity among clusters

It will give you a value between 1 and -1, with values close to 1
indicating well-formed clusters.

If you have labels in your training data, you can also use other
metrics, such as homogenity and completeness, which you will see later
in the lab.

Clustering algorithms are used to tackle many different tasks such as
finding similar users, songs, or images, detecting key trends and
changes in patterns, understanding community structures in social
networks.



Commonly used clustering algorithms
===================================

There are two types of commonly used clustering algorithms:
distance-based and probabilistic models. For example, k-means and
**Density-Based Spatial Clustering of Applications with Noise**
(**DBSCAN**) are distance-based algorithms, whereas the Gaussian mixture
model is probabilistic.

Distance-based algorithms may use a variety of distance measures where
Euclidean distance metrics are usually used.

Probabilistic algorithms will assume that there is a generative process
with a mixture of probability distributions with unknown parameters and
the goal is to calculate these parameters from the data.

Since there are many clustering algorithms, picking the right one
depends on the characteristics of your data. For example, k-means will
work with centroids of clusters and this requires clusters in your data
to be evenly sized and convexly shaped. This means that k-means will not
work well on elongated clusters or irregularly shaped manifolds. When
your clusters in your data are not evenly sized or convexly shaped, you
many want to use DBSCAN to cluster areas of any shape.

Knowing a thing or two about your data will bring you closer to finding
the right algorithms, but what if you don\'t know much about your data?
Many times when you are performing exploratory analysis, it might be
hard to get your head around what\'s happening. If you find yourself in
this kind of situation, an automated unsupervised ML pipeline can help
you to understand the characteristics of your data better.

Be careful when you perform this kind of analysis, though; the actions
you will take later will be driven by the results you will see and this
could quickly send you down the wrong path if you are not cautious.



Creating sample datasets with sklearn
=====================================

In [sklearn], there are some useful ways to create sample datasets
for testing algorithms:


```
# Importing necessary libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set context helps you to adjust things like label size, lines and various elements
# Try "notebook", "talk" or "paper" instead of "poster" to see how it changes
sns.set_context('poster')

# set_color_codes will affect how colors such as 'r', 'b', 'g' will be interpreted
sns.set_color_codes()

# Plot keyword arguments will allow you to set things like size or line width to be used in charts.
plot_kwargs = {'s': 10, 'linewidths': 0.1}

import numpy as np
import pandas as pd

# Pprint will better output your variables in console for readability
from pprint import pprint


# Creating sample dataset using sklearn samples_generator
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Make blobs will generate isotropic Gaussian blobs
# You can play with arguments like center of blobs, cluster standard deviation
centers = [[2, 1], [-1.5, -1], [1, -1], [-2, 2]]
cluster_std = [0.1, 0.1, 0.1, 0.1]

# Sample data will help you to see your algorithms behavior
X, y = make_blobs(n_samples=1000,
                  centers=centers,
                  cluster_std=cluster_std,
                  random_state=53)


# Plot generated sample data
plt.scatter(X[:, 0], X[:, 1], **plot_kwargs)
plt.show()
```


We get the following plot from the preceding code:


![](./images/c732f54a-9bbc-4958-aeb3-f0d68099d2f2.png)


[cluster\_std] will affect the amount of dispersion. Change it to
[\[0.4, 0.5, 0.6, 0.5\]] and try again:


```
cluster_std = [0.4, 0.5, 0.6, 0.5] 

X, y = make_blobs(n_samples=1000,
                  centers=centers,
                  cluster_std=cluster_std,
                  random_state=53)

plt.scatter(X[:, 0], X[:, 1], **plot_kwargs)
plt.show()
```


We get the following plot from the preceding code:


![](./images/c1af55f2-36f3-4e21-a9cc-f8b8269c8a84.png)


Now it looks more realistic!

Let\'s write a small class with helpful methods to create unsupervised
experiments. First, you will use the [fit\_predict] method to
apply one or more clustering algorithms on the sample dataset:


```
class Unsupervised_AutoML:

    def __init__(self, estimators=None, transformers=None):
        self.estimators = estimators
        self.transformers = transformers
        pass
```


[Unsupervised\_AutoML] class will initialize with a set of
estimators and transformers. The second class method will be
[fit\_predict]:


```
def fit_predict(self, X, y=None):
    """
    fit_predict will train given estimator(s) and predict cluster membership for each sample
    """

    # This dictionary will hold predictions for each estimator
    predictions = []
    performance_metrics = {}

    for estimator in self.estimators:
        labels = estimator['estimator'](*estimator['args'], **estimator['kwargs']).fit_predict(X)
        estimator['estimator'].n_clusters_ = len(np.unique(labels))
        metrics = self._get_cluster_metrics(estimator['estimator'].__name__, estimator['estimator'].n_clusters_, X, labels, y)
        predictions.append({estimator['estimator'].__name__: labels})
        performance_metrics[estimator['estimator'].__name__] = metrics

    self.predictions = predictions
    self.performance_metrics = performance_metrics

    return predictions, performance_metrics
```


The [fit\_predict] method uses the [\_get\_cluster\_metrics]
method to get the performance metrics, which is defined in the following
code block:


```
# Printing cluster metrics for given arguments
def _get_cluster_metrics(self, name, n_clusters_, X, pred_labels, true_labels=None):
    from sklearn.metrics import homogeneity_score, \
        completeness_score, \
        v_measure_score, \
        adjusted_rand_score, \
        adjusted_mutual_info_score, \
        silhouette_score

    print("""################## %s metrics #####################""" % name)
    if len(np.unique(pred_labels)) >= 2:

        silh_co = silhouette_score(X, pred_labels)

        if true_labels is not None:

            h_score = homogeneity_score(true_labels, pred_labels)
            c_score = completeness_score(true_labels, pred_labels)
            vm_score = v_measure_score(true_labels, pred_labels)
            adj_r_score = adjusted_rand_score(true_labels, pred_labels)
            adj_mut_info_score = adjusted_mutual_info_score(true_labels, pred_labels)

            metrics = {"Silhouette Coefficient": silh_co,
                       "Estimated number of clusters": n_clusters_,
                       "Homogeneity": h_score,
                       "Completeness": c_score,
                       "V-measure": vm_score,
                       "Adjusted Rand Index": adj_r_score,
                       "Adjusted Mutual Information": adj_mut_info_score}

            for k, v in metrics.items():
                print("\t%s: %0.3f" % (k, v))

            return metrics

        metrics = {"Silhouette Coefficient": silh_co,
                   "Estimated number of clusters": n_clusters_}

        for k, v in metrics.items():
            print("\t%s: %0.3f" % (k, v))

        return metrics

    else:
        print("\t# of predicted labels is {}, can not produce metrics. \n".format(np.unique(pred_labels)))
```


The [\_get\_cluster\_metrics] method calculates metrics, such as
[homogeneity\_score], [completeness\_score],
[v\_measure\_score], [adjusted\_rand\_score],
[adjusted\_mutual\_info\_score], and [silhouette\_score].
These metrics will help you to assess how well the clusters are
separated and also measure the similarity within and between clusters.



K-means algorithm in action
===========================

You can now apply the [KMeans] algorithm to see how it works:


```
from sklearn.cluster import KMeans

estimators = [{'estimator': KMeans, 'args':(), 'kwargs':{'n_clusters': 4}}]

unsupervised_learner = Unsupervised_AutoML(estimators)
```


You can see the [estimators]:


```
unsupervised_learner.estimators
```


These will output the following:


```
[{'args': (),
 'estimator': sklearn.cluster.k_means_.KMeans,
 'kwargs': {'n_clusters': 4}}]
```


You can now invoke [fit\_predict] to obtain [predictions]
and [performance\_metrics]:


```
predictions, performance_metrics = unsupervised_learner.fit_predict(X, y)
```


Metrics will be written to the console:


```
################## KMeans metrics #####################
  Silhouette Coefficient: 0.631
  Estimated number of clusters: 4.000
  Homogeneity: 0.951
  Completeness: 0.951
  V-measure: 0.951
  Adjusted Rand Index: 0.966
  Adjusted Mutual Information: 0.950
```


You can always print metrics later:


```
pprint(performance_metrics)
```


This will output the name of the estimator and its metrics:


```
{'KMeans': {'Silhouette Coefficient': 0.9280431207593165, 'Estimated number of clusters': 4, 'Homogeneity': 1.0, 'Completeness': 1.0, 'V-measure': 1.0, 'Adjusted Rand Index': 1.0, 'Adjusted Mutual Information': 1.0}}
```


Let\'s add another class method to plot the clusters of the given
estimator and predicted labels:


```
# plot_clusters will visualize the clusters given predicted labels
def plot_clusters(self, estimator, X, labels, plot_kwargs):

    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

    plt.scatter(X[:, 0], X[:, 1], c=colors, **plot_kwargs)
    plt.title('{} Clusters'.format(str(estimator.__name__)), fontsize=14)
    plt.show()


```


Let\'s see the usage:


```
plot_kwargs = {'s': 12, 'linewidths': 0.1}
unsupervised_learner.plot_clusters(KMeans,
                                   X,
                                   unsupervised_learner.predictions[0]['KMeans'],
                                   plot_kwargs)
```


You get the following plot from the preceding block:


![](./images/3591f881-f418-4f3f-a586-9a4cd8dc535c.png)


In this example, clusters are evenly sized and clearly separate from
each other but, when you are doing this kind of exploratory analysis,
you should try different hyperparameters and examine the results.

You will write a wrapper function later in this lab to apply a list
of clustering algorithms and their hyperparameters to examine the
results. For now, let\'s see one more example with k-means where it does
not work well.

When clusters in your dataset have different statistical properties,
such as differences in variance, k-means will fail to identify clusters
correctly:


```
X, y = make_blobs(n_samples=2000, centers=5, cluster_std=[1.7, 0.6, 0.8, 1.0, 1.2], random_state=220)

# Plot sample data
plt.scatter(X[:, 0], X[:, 1], **plot_kwargs)
plt.show()
```


We get the following plot from the preceding code:


![](./images/ba3f5565-737a-4519-bea1-b1a7ab202a06.png)


Although this sample dataset is generated with five centers, it\'s not
that obvious and there might be four clusters, as well:


```
from sklearn.cluster import KMeans

estimators = [{'estimator': KMeans, 'args':(), 'kwargs':{'n_clusters': 4}}]

unsupervised_learner = Unsupervised_AutoML(estimators)

predictions, performance_metrics = unsupervised_learner.fit_predict(X, y)


```


Metrics in the console are as follows:


```
################## KMeans metrics #####################
  Silhouette Coefficient: 0.549
  Estimated number of clusters: 4.000
  Homogeneity: 0.729
  Completeness: 0.873
  V-measure: 0.795
  Adjusted Rand Index: 0.702
  Adjusted Mutual Information: 0.729
```


[KMeans] clusters are plotted as follows:


```
plot_kwargs = {'s': 12, 'linewidths': 0.1}
unsupervised_learner.plot_clusters(KMeans,
                                   X,
                                   unsupervised_learner.predictions[0]['KMeans'],
                                   plot_kwargs)
```


We get the following plot from the preceding code:


![](./images/0665e4c2-6eb2-44a9-aae4-b1aef376bb28.png)


In this example, points between red (dark gray) and bottom-green
clusters (light gray) seem to form one big cluster. K-means is
calculating the centroid based on the mean value of points surrounding
that centroid. Here, you need to have a different approach.



The DBSCAN algorithm in action
==============================

DBSCAN is one of the clustering algorithms that can deal with non-flat
geometry and uneven cluster sizes. Let\'s see what it can do:


```
from sklearn.cluster import DBSCAN

estimators = [{'estimator': DBSCAN, 'args':(), 'kwargs':{'eps': 0.5}}]

unsupervised_learner = Unsupervised_AutoML(estimators)

predictions, performance_metrics = unsupervised_learner.fit_predict(X, y)
```


Metrics in the console are as follows:


```
################## DBSCAN metrics #####################
  Silhouette Coefficient: 0.231
  Estimated number of clusters: 12.000
  Homogeneity: 0.794
  Completeness: 0.800
  V-measure: 0.797
  Adjusted Rand Index: 0.737
  Adjusted Mutual Information: 0.792
```


[DBSCAN] clusters are plotted as follows:


```
plot_kwargs = {'s': 12, 'linewidths': 0.1}
unsupervised_learner.plot_clusters(DBSCAN,
                                   X,
                                   unsupervised_learner.predictions[0]['DBSCAN'],
                                   plot_kwargs)
```


We get the following plot from the preceding code:


![](./images/d0570eb3-7bae-4ff5-89de-a7867e55216f.png)


Conflict between red (dark gray) and bottom-green (light gray) clusters
from the k-means case seems to be gone, but what\'s interesting here is
that some small clusters appeared and some points were not assigned to
any cluster based on their distance.

DBSCAN has the [eps(epsilon)] hyperparameter, which is related to
proximity for points to be in same neighborhood; you can play with that
parameter to see how the algorithm behaves.

When you are doing this kind of exploratory analysis where you don\'t
know much about the data, visual clues are always important, because
metrics can mislead you since not every clustering algorithm can be
assessed using similar metrics.



Agglomerative clustering algorithm in action
============================================

Our last try will be with an agglomerative clustering algorithm:


```
from sklearn.cluster import AgglomerativeClustering

estimators = [{'estimator': AgglomerativeClustering, 'args':(), 'kwargs':{'n_clusters': 4, 'linkage': 'ward'}}]

unsupervised_learner = Unsupervised_AutoML(estimators)

predictions, performance_metrics = unsupervised_learner.fit_predict(X, y)
```


Metrics in the console are as follows:


```
################## AgglomerativeClustering metrics #####################
  Silhouette Coefficient: 0.546
  Estimated number of clusters: 4.000
  Homogeneity: 0.751
  Completeness: 0.905
  V-measure: 0.820
  Adjusted Rand Index: 0.719
  Adjusted Mutual Information: 0.750
```


[AgglomerativeClustering] clusters are plotted as follows:


```
plot_kwargs = {'s': 12, 'linewidths': 0.1}
unsupervised_learner.plot_clusters(AgglomerativeClustering,
                                   X,
                                   unsupervised_learner.predictions[0]['AgglomerativeClustering'],
                                   plot_kwargs)
```


We get the following plot from the preceding code:


![](./images/042a646b-2906-493d-b800-a0950d583bbe.png)


[AgglomerativeClustering] behaved like k-means in this example,
with slight differences.


Visualizing high-dimensional datasets
=====================================

What about visually inspecting datasets that have more than three
dimensions? In order to visually inspect your dataset, you need to have
a maximum of three dimensions; if not, you need to use specific methods
to reduce dimensionality. This is usually achieved by applying a
**Principal Component Analysis** (**PCA**) or t-SNE algorithm.

The following code will load the [Breast Cancer Wisconsin
Diagnostic] dataset, which is commonly used in ML tutorials:


```
# Wisconsin Breast Cancer Diagnostic Dataset
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
X = data.data

df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()
```


Output in the console is as follows:


```
   mean radius mean texture mean perimeter mean area mean smoothness \
0 17.99 10.38 122.80 1001.0 0.11840 
1 20.57 17.77 132.90 1326.0 0.08474 
2 19.69 21.25 130.00 1203.0 0.10960 
3 11.42 20.38 77.58 386.1 0.14250 
4 20.29 14.34 135.10 1297.0 0.10030 

...

   mean fractal dimension ... worst radius \
0 0.07871 ... 25.38 
1 0.05667 ... 24.99 
2 0.05999 ... 23.57 
3 0.09744 ... 14.91 
4 0.05883 ... 22.54 

...

   worst fractal dimension 
0 0.11890 
1 0.08902 
2 0.08758 
3 0.17300 
4 0.07678 
```


You have thirty different features to use to understand the different
characteristics of a tumor in the given patient.

[df.describe()] will show you descriptive statistics for each
feature:


```
df.describe()

       mean radius mean texture mean perimeter mean area \
count 569.000000 569.000000 569.000000 569.000000 
mean 14.127292 19.289649 91.969033 654.889104 
std 3.524049 4.301036 24.298981 351.914129 
min 6.981000 9.710000 43.790000 143.500000 
25% 11.700000 16.170000 75.170000 420.300000 
50% 13.370000 18.840000 86.240000 551.100000 
75% 15.780000 21.800000 104.100000 782.700000 
max 28.110000 39.280000 188.500000 2501.000000 

...

       mean symmetry mean fractal dimension ... \
count 569.000000 569.000000 ... 
mean 0.181162 0.062798 ... 
std 0.027414 0.007060 ... 
min 0.106000 0.049960 ... 
25% 0.161900 0.057700 ... 
50% 0.179200 0.061540 ... 
75% 0.195700 0.066120 ... 
max 0.304000 0.097440 ... 

...

       worst concave points worst symmetry worst fractal dimension 
count 569.000000 569.000000 569.000000 
mean 0.114606 0.290076 0.083946 
std 0.065732 0.061867 0.018061 
min 0.000000 0.156500 0.055040 
25% 0.064930 0.250400 0.071460 
50% 0.099930 0.282200 0.080040 
75% 0.161400 0.317900 0.092080 
max 0.291000 0.663800 0.207500 
[8 rows x 30 columns]
```


Let\'s see the results before and after scaling. The following code
snippet will fit the PCA to the original data.



Principal Component Analysis in action
======================================

The following code block shows you how to apply PCA with two components
and visualize the results:


```
# PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2, whiten=True)
pca = pca.fit_transform(df)

plt.scatter(pca[:, 0], pca[:, 1], c=data.target, cmap="RdBu_r", edgecolor="Red", alpha=0.35)
plt.colorbar()
plt.title('PCA, n_components=2')
plt.show()
```


We get the following plot from the preceding code:


![](./images/e35afdf5-6536-4a7f-be4b-0bbde1b7177c.png)



Plot of PCA, n\_components=2


Here, you can see the red class (dark gray) is very condensed into one
particular area and it\'s hard to separate classes. Differences in
variances distort our view and scaling can help:


```
# Preprocess data.
scaler = StandardScaler()
scaler.fit(df)
preprocessed_data = scaler.transform(df)
scaled_features_df = pd.DataFrame(preprocessed_data, index=df.index, columns=df.columns)
```


After preprocessing data by applying [StandardScaler], the dataset
has unit variance:


```
scaled_features_df.describe()

        mean radius mean texture mean perimeter mean area \
count 5.690000e+02 5.690000e+02 5.690000e+02 5.690000e+02 
mean -3.162867e-15 -6.530609e-15 -7.078891e-16 -8.799835e-16 
std 1.000880e+00 1.000880e+00 1.000880e+00 1.000880e+00 
min -2.029648e+00 -2.229249e+00 -1.984504e+00 -1.454443e+00 
25% -6.893853e-01 -7.259631e-01 -6.919555e-01 -6.671955e-01 
50% -2.150816e-01 -1.046362e-01 -2.359800e-01 -2.951869e-01 
75% 4.693926e-01 5.841756e-01 4.996769e-01 3.635073e-01 
max 3.971288e+00 4.651889e+00 3.976130e+00 5.250529e+00 

...
 
       mean symmetry mean fractal dimension ... \
count 5.690000e+02 5.690000e+02 ... 
mean -1.971670e-15 -1.453631e-15 ... 
std 1.000880e+00 1.000880e+00 ... 
min -2.744117e+00 -1.819865e+00 ... 
25% -7.032397e-01 -7.226392e-01 ... 
50% -7.162650e-02 -1.782793e-01 ... 
75% 5.307792e-01 4.709834e-01 ... 
max 4.484751e+00 4.910919e+00 ... 

...

       worst concave points worst symmetry worst fractal dimension 
count 5.690000e+02 5.690000e+02 5.690000e+02 
mean -1.412656e-16 -2.289567e-15 2.575171e-15 
std 1.000880e+00 1.000880e+00 1.000880e+00 
min -1.745063e+00 -2.160960e+00 -1.601839e+00 
25% -7.563999e-01 -6.418637e-01 -6.919118e-01 
50% -2.234689e-01 -1.274095e-01 -2.164441e-01 
75% 7.125100e-01 4.501382e-01 4.507624e-01 
max 2.685877e+00 6.046041e+00 6.846856e+00 
[8 rows x 30 columns]
```


Apply PCA to see whether the first two principal components are enough
to separate labels:


```
# PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2, whiten=True)
pca = pca.fit_transform(scaled_features_df)

plt.scatter(pca[:, 0], pca[:, 1], c=data.target, cmap="RdBu_r", edgecolor="Red", alpha=0.35)
plt.colorbar()
plt.title('PCA, n_components=2')
plt.show()
```


We get the following output from the preceding code:


![](./images/6d0e6efc-0ed8-432b-9e47-15cd3b0dab5d.png)



PCA, n\_components=2, after scaling


This seems interesting, as examples with different labels are mostly
separated using the first two principal components.



t-SNE in action
===============

You can also try t-SNE to visualize high-dimensional data. First,
[TSNE] will be applied to the original data:


```
# TSNE
from sklearn.manifold import TSNE

tsne = TSNE(verbose=1, perplexity=40, n_iter=4000)
tsne = tsne.fit_transform(df)
```


Output in the console is as follows:


```
[t-SNE] Computing 121 nearest neighbors...
[t-SNE] Indexed 569 samples in 0.000s...
[t-SNE] Computed neighbors for 569 samples in 0.010s...
[t-SNE] Computed conditional probabilities for sample 569 / 569
[t-SNE] Mean sigma: 33.679703
[t-SNE] KL divergence after 250 iterations with early exaggeration: 48.886528
[t-SNE] Error after 1600 iterations: 0.210506
```


Plotting the results is as follows:


```
plt.scatter(tsne[:, 0], tsne[:, 1], c=data.target, cmap="winter", edgecolor="None", alpha=0.35)
plt.colorbar()
plt.title('t-SNE')
plt.show()
```


We get the following output from the preceding code:


![](./images/fd0133e2-c27f-4964-b7ec-bea65b5c6552.png)



Plot of TSNE


Applying [TSNE] on scaled data is as follows:


```
tsne = TSNE(verbose=1, perplexity=40, n_iter=4000)
tsne = tsne.fit_transform(scaled_features_df)
```


Output in the console is as follows:


```
[t-SNE] Computing 121 nearest neighbors...
[t-SNE] Indexed 569 samples in 0.001s...
[t-SNE] Computed neighbors for 569 samples in 0.018s...
[t-SNE] Computed conditional probabilities for sample 569 / 569
[t-SNE] Mean sigma: 1.522404
[t-SNE] KL divergence after 250 iterations with early exaggeration: 66.959343
[t-SNE] Error after 1700 iterations: 0.875110
```


Plotting the results is as follows:


```
plt.scatter(tsne[:, 0], tsne[:, 1], c=data.target, cmap="winter", edgecolor="None", alpha=0.35)
plt.colorbar()
plt.title('t-SNE')
plt.show()
```


We get the following output from the preceding code:


![](./images/5bb644e3-ff59-46fc-8783-ccec1c62f0ba.png)



Summary
=======

In this lab, you learned about many different aspects when it comes
to choosing a suitable ML pipeline for a given problem.

Computational complexity, differences in training and scoring time,
linearity versus non-linearity, and algorithm, specific feature
transformations are valid considerations and it's useful to look at your
data from these perspectives.

You gained a better understanding of selecting suitable models and how
machine learning pipelines work by practicing various use cases. You are
starting to scratch the surface and this lab was a good starting
point to extend these skills.
