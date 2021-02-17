# Udacity-Capstone-Project-Stock-Price-Predictor



## Overview of the Project
This is my submission of Udacity's Investment and Trading Capstone Project for the Data Scientist Nano Degree. The purpose of this project is to create a easy-to-use python notebook where predictions about stock prices can be made. The main features are:
  - Almost Unlimited Pool of Stocks for predictions: The data for this project is downloaded from Yahoo! Finance using the pandas_datareader library. As such, almost any stock with historical price data that is available on that site can be predicted. The exception being stocks with little history. 
  - Many Stocks can be predicted at once: The user can pass a list of stock tickers (as long as it can be) and all of them will be predicted
  - Almost any date for predictions: The user can input a date called target_date and the model will output the stock price prediction for that particular date. This works because the model's training and prediction occur at the same time meaning that, whenever a query is made, the model trains itself to fit the characteristics of the query. This means that focus on speed is paramount and some things such as cross validation are not built in.

## This Repository

The repository holds two files:
  - stock_predict_lib.py
  - Stock_Price_Prediction_Notebook.ipynb

The first file holds all the functions written and used in the project. The second file is a notebook showing how to use these function to train the machine learning models and use them to predict the stock prices. The steps a user would follow if he/she wants to use the project are:
  - Input a list of stock tickers, a start and an end date for the training of the models (This should be sufficiently long. 1-2 years is recommended), and a target date greater than the end_date.
  - Run the functions belonging to the 'Training Module' (as seen in the diagram below). As of now, the user must run each function separately.
  - (OPTIONAL) Plot the perfomance of the models on the test set.
  - Predict the stock prices on the target date.

This steps are better understood by looking at the project's general diagram where spl. denotes the stock_predict_lib module

![General Diagram](https://user-images.githubusercontent.com/46632664/108224169-9c3f4000-7108-11eb-8ce1-f5f0f5f28af1.png)
