# Stock Price Prediction [[Code]](https://github.com/SarthakPatidar/Machine-Learning/blob/master/Supervised%20Learning/projects/stock%20price%20prediction/stock_prediction.py)

## Dataset Used
* [Quandl](https://quandl.com) - WIKI/GOOGL

## Algorithm Implemented
* [Linear Regression](https://github.com/SarthakPatidar/Machine-Learning/blob/master/Supervised%20Learning/algorithms/linear%20regression/doc/README.md)

## Steps Followed
### 1. Building Features 
* Indentify the relevant features.
* Calculate the potential volatility by using percent changes.
* Filter out the relevant features.

### 2. Building Label
* Fill in the missing values.
* Add Label column having close price of past 1 % of data

### 3. Training and Testing Dataset
* Scaling the features in range [-1, 1] (using sklearn --> preprocessing)
* Shuffling the data into training testing test (using sklearn.model_selection import        
  train_test_split)
* Import the Regression Model [LinearRegression and SVM] (using sklearn --> svm & sklearn. 
  linear_model --> LinearRegression)
* Features (X) : Drop the label column from dataframe and set it to X
* Preprocess X to normalize values.
* Set X till the forecast_out row 
* Set X_Lately consisting of the rows with NAN forecast_col entries.
* Label (y) : The 'label' column of dataframe
* Train the data using LinearRegression / SVM
* Obtain the score of prediction

### 4. Plotting and Charting
* Import matplotlib.pyplot
* Build a forecast column
* Plot the Adj. Close and Forecast column against the Date.

### 5. Pickling and Scaling
* Pickle - Saving a classifier to a point so that we do not train everytime we run the tests.
* Open LinearRegression.pickle and dump the classifier.
* Read the pickle and load the classifier.

## Predicted Stock Price Trend
![Image of Trend](https://github.com/SarthakPatidar/Machine-Learning/blob/master/Supervised%20Learning/projects/stock%20price%20prediction/resources/pred2.png)

![Image of Prediction](https://github.com/SarthakPatidar/Machine-Learning/blob/master/Supervised%20Learning/projects/stock%20price%20prediction/resources/pred1.png)
