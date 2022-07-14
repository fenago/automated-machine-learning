
Lab 1: Introduction to AutoML
=============================

In this course, we will be covering the following topics:

-   Why use AutoML and how it helps
-   When to use AutoML
-   Overview of AutoML libraries


#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

All notebooks are present in `lab 01` folder.


A k-means algorithm helps to cluster similar data points together. The
following code snippet uses the scikit-learn library, don\'t
worry if you don\'t understand every line:


```
# Sklearn has convenient modules to create sample data.
# make_blobs will help us to create a sample data set suitable for clustering
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.30, random_state=0)

# Let's visualize what we have first
import matplotlib.pyplot as plt
import seaborn as sns

plt.scatter(X[:, 0], X[:, 1], s=50)
```


The output of the preceding code snippet is as follows:


![](./images/dde330f1-f83c-4e78-9a80-0bf8deeeecb1.png)


You can easily see that we have two clusters on the plot:


```
# We will import KMeans model from clustering model family of Sklearn
from sklearn.cluster import KMeans

k_means = KMeans(n_clusters=2)
k_means.fit(X)
predictions = k_means.predict(X)

# Let's plot the predictions
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='brg')
```


The output of the preceding code snippet is as follows:


![](./images/5509acef-8b5b-415a-b667-a4d94fac262d.png)


Nice! Our algorithm worked as we expected. Astute readers may have
noticed that there was an argument named [n\_clusters] for the
k-means model. When you provide this value to the k-means algorithm, it
will try to split this dataset into two clusters. As you can guess,
k-means\'s hyperparameter in this case is the number of clusters. The
k-means model needs to know this parameter before training.

Different algorithms have different hyperparameters such as depth of
tree for decision trees, number of hidden layers, learning rate for
neural networks, alpha parameter for Lasso or C, kernel, and gamma for
**Support Vector Machines** (**SVMs**).

Let\'s see how many arguments the k-means model has by using the
[get\_params] method:


```
k_means.get_params()
```


The output will be the list of all parameters that you can optimize:


```
{'algorithm': 'auto',
 'copy_x': True,
 'init': 'k-means++',
 'max_iter': 300,
 'n_clusters': 2,
 'n_init': 10,
 'n_jobs': 1,
 'precompute_distances': 'auto',
 'random_state': None,
 'tol': 0.0001,
 'verbose': 0}
```


In most real-life use cases, you will neither have resources nor time
for trying each possible combination with the options of all steps
considered.

AutoML libraries come to your aid at this point by carefully setting up
experiments for various ML pipelines, which covers all the steps from
data ingestion, data processing, modeling, and scoring.


What will you learn?
====================

Throughout this course, you will learn both theoretical and practical
aspects of AutoML systems. More importantly, you will practice your
skills by developing an AutoML system from scratch.



Overview of AutoML libraries
============================

There are many popular AutoML libraries, and in this section you will
have an overview of commonly used ones in the data science community.



Featuretools
============

Featuretools (<https://www.featuretools.com/>) is a good library for
automatically engineering features from relational and transactional
data. The library introduces the concept called **Deep Feature
Synthesis** (**DFS**). If you have multiple datasets with relationships
defined among them such as parent-child based on columns that you use as
unique identifiers for examples, DFS will create new features based on
certain calculations, such as summation, count, mean, mode, standard
deviation, and so on. Let\'s go through a small example where you will
have two tables, one showing the database information and the other
showing the database transactions for each database:


```
import pandas as pd

# First dataset contains the basic information for databases.
databases_df = pd.DataFrame({"database_id": [2234, 1765, 8796, 2237, 3398], 
"creation_date": ["2018-02-01", "2017-03-02", "2017-05-03", "2013-05-12", "2012-05-09"]})

databases_df.head()
```


You get the following output:


![](./images/ab77bbb6-073a-4c39-8f2e-c886f5b20439.png)


The following is the code for the database transaction:


```
# Second dataset contains the information of transaction for each database id
db_transactions_df = pd.DataFrame({"transaction_id": [26482746, 19384752, 48571125, 78546789, 19998765, 26482646, 12484752, 42471125, 75346789, 16498765, 65487547, 23453847, 56756771, 45645667, 23423498, 12335268, 76435357, 34534711, 45656746, 12312987], 
                "database_id": [2234, 1765, 2234, 2237, 1765, 8796, 2237, 8796, 3398, 2237, 3398, 2237, 2234, 8796, 1765, 2234, 2237, 1765, 8796, 2237], 
                "transaction_size": [10, 20, 30, 50, 100, 40, 60, 60, 10, 20, 60, 50, 40, 40, 30, 90, 130, 40, 50, 30],
                "transaction_date": ["2018-02-02", "2018-03-02", "2018-03-02", "2018-04-02", "2018-04-02", "2018-05-02", "2018-06-02", "2018-06-02", "2018-07-02", "2018-07-02", "2018-01-03", "2018-02-03", "2018-03-03", "2018-04-03", "2018-04-03", "2018-07-03", "2018-07-03", "2018-07-03", "2018-08-03", "2018-08-03"]})

db_transactions_df.head()
```


You get the following output:


![](./images/3d585108-1615-44a7-a2cb-faaf3785da99.png)


The code for the entities is as follows:


```
# Entities for each of datasets should be defined
entities = {
"databases" : (databases_df, "database_id"),
"transactions" : (db_transactions_df, "transaction_id")
}

# Relationships between tables should also be defined as below
relationships = [("databases", "database_id", "transactions", "database_id")]

print(entities)


```


You get the following output for the preceding code:


![](./images/ed0cbaba-10c1-4203-abd0-57beac1876d1.png)



TPOT
====

**Tree-Based Pipeline Optimization Tool** (**TPOT**) is using genetic
programming to find the best performing ML pipelines, and it is built on
top of scikit-learn.

Once your dataset is cleaned and ready to be used, TPOT will help you
with the following steps of your ML pipeline:

-   Feature preprocessing
-   Feature construction and selection
-   Model selection
-   Hyperparameter optimization

Once TPOT is done with its experimentation, it will provide you with the
best performing pipeline.

TPOT is very user-friendly as it\'s similar to using scikit-learn\'s
API:


```
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Digits dataset that you have used in Auto-sklearn example
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

# You will create your TPOT classifier with commonly used arguments
tpot = TPOTClassifier(generations=10, population_size=30, verbosity=2)

# When you invoke fit method, TPOT will create generations of populations, seeking best set of parameters. Arguments you have used to create TPOTClassifier such as generations and population_size will affect the search space and resulting pipeline.
tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))
# 0.9834
tpot.export('my_pipeline.py')
```


Once you have exported your pipeline in the Python
[my\_pipeline.py] file, you will see the selected pipeline
components:


```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target =\
            train_test_split(features, tpot_data['target'].values, random_state=42)


exported_pipeline = KNeighborsClassifier(n_neighbors=6, 
   weights="distance")

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)


```



Summary
=======

You have reviewed the core components of AutoML systems and also
practiced your skills using popular AutoML libraries.
