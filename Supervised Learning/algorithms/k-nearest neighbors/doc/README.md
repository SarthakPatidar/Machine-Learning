# K-Nearest Neighbors [[Code]](https://github.com/SarthakPatidar/Machine-Learning/blob/master/Supervised%20Learning/algorithms/k-nearest%20neighbors/k_nearest_neighbor.py)

## Objectives
* Classify the data point into particular group based on value of K (number of neighbors).

## Steps Followed
* Defining the dataset : Create a dictionary with key as the category and value as list of features.
* Define the test data point to be classified.
* Calculate the Euclidean distance of the test data point from every other data point in dataset.
* Find the category of the K data points in dataset having minimum euclidean distance from test data point.
* The category of test data point = mode(category of K data points)

## Terminologies 
* Confidence : Determined by how well does the classifier assigns a group to a particular data point. 
* Accuracy : Determined by how well the classifier performs on the testing dataset.
* Euclidean Distance : The distance between the data points calculated by the formula :  <br/><br/> <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/795b967db2917cdde7c2da2d1ee327eb673276c0" alt="euclidean_distance">

## Best Practices
* Pick a value of K to be an odd number greater than number of groups.

## References 
* The euclidean distance can be calculated using [np.linalg.norm()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html)
