# Udacity-Capstone-Project-Stock-Price-Predictor

## Medium Link
This is the project as seen on Medium https://smurguersons.medium.com/stock-price-prediction-with-python-168925c5ceb7

## Overview of the Project
This is my submission of Udacity's Investment and Trading Capstone Project for the Data Scientist Nano Degree. The purpose of this project is to create a easy-to-use python notebook and function module where predictions about stock prices can be made. The main features are:
  - The data for this project is historical adjusted closing prices for stocks. They are downloaded from Yahoo! Finance using the pandas_datareader library. As such, almost any stock with historical price data that is available on that site can be predicted. The exception being stocks with little history (less than 6 months). Also, the user can pass a list of stock tickers (as long as the user wishes) and all of them will be trained on and predicted.
  - Here, the user has all the power on deciding what data should the models be built upon: Do they want to train on pre-covid historical data in order to avoid those downturns? They can. Do they want to train on data between 2010 and 2015? They can. The interval for training is governed by two parameters called start_date and end_date which are the starting and final dates for which data will be gathered to create the training dataset (See Methodology section).
  - The user inputs a date called target_date and the model will output the stock price prediction for that particular date. This works because the model’s training and prediction occur at the same time meaning that, whenever a query is made, the model trains itself to fit the characteristics of the query. This means that focus on speed is paramount and some features such as cross validation are not built in. The only requirement is that this target_date should be greater than another input called end_date.
  - Once trained, the model can be used to predict stock prices on any point between the end_date and the target_date without needing to be retrained. Thus if the end_date was January 1st 2021 and the target_date was March 1st 2021, the model can be used to predict the stock prices at any point between these two dates without needing to be retrained.

As a measure of the speed of the project, a complete pass from input to prediction for 3 stocks on 1 year data takes around 3 minutes to complete. with most of the time being spent on feature extraction.

## Libraries Used
The following libraries were used

  - pandas_datareader
  - matplotlib
  - seaborn
  - pandas
  - numpy
  - datetime
  - tsfresh
  - sklearn

## This Repository

The repository holds two files:
  - stock_predict_lib.py
  - Stock_Price_Prediction_Notebook.ipynb

The first file holds all the functions written and used in the project. The second file is a notebook showing how to use these function to train the machine learning models and use them to predict the stock prices. The steps a user would follow if he/she wants to use the project are:
  - Input a list of stock tickers, a start and an end date for the training of the models (This should be sufficiently long. 1-2 years is recommended), and a target date greater than the end_date.
  - Run the functions belonging to the 'Training Module' (as seen in the diagram below). As of now, the user must run each function separately.
  - (OPTIONAL) Plot the perfomance of the models on the test set.
  - Predict the stock prices on the target date.

These steps are better understood by looking at the project's general diagram where spl. denotes the stock_predict_lib module. All of them are written in the file Stock_Price_Prediction_Notebook.ipynb.

![General Diagram](https://user-images.githubusercontent.com/46632664/108224169-9c3f4000-7108-11eb-8ce1-f5f0f5f28af1.png)

## Results

The results of the project are dependent on the user's choice of stocks and period to study. However, for model selection purposes, the metric R2 was used. In the example shown in the medium post, the following results were obtained

![r2](https://user-images.githubusercontent.com/46632664/108282486-b51f1400-714f-11eb-915e-f893a2f3e388.PNG)

As it's discussed on the medium article, a model with a R2 metric greater than 95% is chosen as 'ready to use' whereas models with less than 95% are studied on a case by case basis (See the **Evaluation** Discussion on Medium). The predictions for the 3 stocks were done for Febrary 25th 2021, these are shown below

![prediction](https://user-images.githubusercontent.com/46632664/108282540-cd8f2e80-714f-11eb-8892-0aece1f36754.PNG)

