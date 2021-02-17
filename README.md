# Udacity-Capstone-Project-Stock-Price-Predictor



## Overview of the Project
This is my submission of Udacity's Investment and Trading Capstone Project for the Data Scientist Nano Degree. The purpose of this project is to create a easy-to-use python notebook where predictions about stock prices can be made. The main features are:
  - The data for this project is downloaded from Yahoo! Finance using the pandas_datareader library. As such, almost any stock with historical price data that is available on that site can be predicted. The exception being stocks with little history. Also, the user can pass a list of stock tickers (as long as the user wishes) and all of them will be predicted
  - The user can input a date called target_date and the model will output the stock price prediction for that particular date. This works because the model's training and prediction occur at the same time meaning that, whenever a query is made, the model trains itself to fit the characteristics of the query. This means that focus on speed is paramount and some things such as cross validation are not built in. The only requeriment is that this target_date should be greater than another input called end_date which is the final date for which data will be gathered to create the training dataset.
  - Once trained the model be used to predict stock prices on any date between the end_date and the target_date without needing to be retrained. Thus if the end_date was January 1st 2021 and the target_date was March 1st 2021, the model can be used to predict the stock prices at any point between these two dates without needing to be retrained. 

Even with all these features, the use of stablished libraries such as pandas_datareader and tsfresh allows the project to be fast, with a complete pass from input to prediction for 3 stocks on 1 year data taking around 3 minutes to complete with most of the time being spent on feautre extraction.

## This Repository

The repository holds two files:
  - stock_predict_lib.py
  - Stock_Price_Prediction_Notebook.ipynb

The first file holds all the functions written and used in the project. The second file is a notebook showing how to use these function to train the machine learning models and use them to predict the stock prices. The steps a user would follow if he/she wants to use the project are:
  - Input a list of stock tickers, a start and an end date for the training of the models (This should be sufficiently long. 1-2 years is recommended), and a target date greater than the end_date.
  - Run the functions belonging to the 'Training Module' (as seen in the diagram below). As of now, the user must run each function separately.
  - (OPTIONAL) Plot the perfomance of the models on the test set.
  - Predict the stock prices on the target date.

This steps are better understood by looking at the project's general diagram where spl. denotes the stock_predict_lib module. All these steps are written in the file Stock_Price_Prediction_Notebook.ipynb.

![General Diagram](https://user-images.githubusercontent.com/46632664/108224169-9c3f4000-7108-11eb-8ce1-f5f0f5f28af1.png)
