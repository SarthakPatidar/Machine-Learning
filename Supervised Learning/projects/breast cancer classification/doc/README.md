## Dataset Used 
* [Breast Cancer Wisconsin (Original) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29)

## Algorithm Implemented
* [K-Nearest Neighbor Classification](https://github.com/SarthakPatidar/Machine-Learning/blob/master/Supervised%20Learning/algorithms/k-nearest%20neighbors/doc/README.md)

## Steps Followed
### Pre-processing the dataset
* Import the data file of the dataset as csv into a dataframe.
* Impute the missing values.
* Drop the unwanted columns from the dataframe.
* Split the training and the testing data.

### Training the Model
* Import neighbors from scikit-learn
* Build a KNeighborClassifier with the default parameters.
* Train the classifier with training data.

### Calculating the Accuracy
* Calculate score on testing data.