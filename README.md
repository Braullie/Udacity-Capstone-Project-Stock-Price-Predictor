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

The first file holds all the functions written and used in the project

![alt text](https://github.com/Braullie/Udacity-Capstone-Project-Stock-Price-Predictor/tree/main/readme_res/General Diagram.png?raw=true)
