from pandas_datareader import data

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from datetime import date
from datetime import datetime
from datetime import timedelta

from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from tsfresh.transformers import RelevantFeatureAugmenter


def download_prices(tickers, start_date, end_date):
  """
  INPUT:
  tickers : list containing the tickers of the stocks whose prices will be predicted
  start_date : initial date to gather data
  end_data : final date to gather data

  OUTPUT:
  prices_base : dataframe containing the adjusted closing price for the stocks
                on the desired time frame
  """
  panel_data = data.DataReader(
      tickers,
      'yahoo',
      start_date,
      end_date)

  panel_data = pd.DataFrame(panel_data)
  prices_base = panel_data['Adj Close']

  prices = prices_base.stack()
  prices = pd.DataFrame(prices)
  prices.columns = ['Prices']

  prices.reset_index(inplace=True)
  prices = prices.sort_values(by = ['Symbols','Date'])
  prices

  return prices


def calc_prices_windows(prices, tickers, target_date, end_date, t = 30):
  """
  This function calculates rolling window time series for each of the stocks based
  on price data. These time series are used to create the training datasets for the stocks

  INPUT:
  prices : Dataframe containing the price data for all the stocks in a stacked format
  tickers : list of all the stock tickers
  target_date : Date of prediction
  end_date : Last date of information. This is put in order to check if the target_date
             is indeed greater than the end_date.
  t : Parameter that dictates how many days are taken in each of the rolling window
      time series

  OUTPUT:
  prices_windows : Dataframe containing the rolling window time series
  prices_predict : Dataframe containing the response variable of the rolling window time series
                   i.e. the price of the stock a delta days after each time series where delta
                        is given by the day difference between the target_day and today
  """

  target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')
  end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

  if target_date_obj <= end_date_obj:
    raise Error("failed because end_date is greater than target_date")

  today = datetime.combine(date.today(), datetime.min.time())

  delta = target_date_obj - today
  delta_days = delta.days

  prices_windows = pd.DataFrame(columns = ['Date', 'Symbols', 'Prices','id'])
  prices_predict = pd.DataFrame(columns = ['id', 'Symbols', 'Price_Target'])

  for ticker in tickers:
    aux_df = prices[prices.Symbols == ticker]

    n = aux_df.shape[0]

    k = n - t + 1 - (t+1)
    
    for i in range(k):
      aux_df_1 = aux_df[i:i+t];
      aux_df_1['id'] = ticker + '_' + str(i)

      aux_df_2 = pd.DataFrame(
          {'id' : ticker + '_' + str(i),
          'Symbols' : ticker,
          'Price_Target' : aux_df.Prices.iloc[i+t+delta_days]
          },
          index=[0]
      )
      
      prices_windows = prices_windows.append(aux_df_1, ignore_index = True)
      prices_predict = prices_predict.append(aux_df_2, ignore_index = True)

  return prices_windows, prices_predict


def pipeline_creation(prices_windows, prices_predict):
  """
  This function creates the pipeline that transforms the rolling window time series
  into actual time series based features ready to be used in the machine learning
  model. It uses RelevantFeatureAugmenter from tsfresh which not only calculates
  the features but also filters them based on their importance and applicability.

  INPUT:
  prices_windows : Collection of rolling window times series for all the stocks 
                   under study.
  prices_predict : Response variables for each of the rolling window time series
                   of the stocks under study.

  OUTPUT:
  pipeline : sklearn pipeline with the information to transform time series into 
             features for use in fitting and prediction of the machine learning model
  price_target_model_df : Training dataset complete with features and response variables.
  """


  pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(
      column_id = 'id', 
      column_sort = 'Date',
      column_value = 'Prices'))])

  prices_predict.set_index('id', inplace = True)

  X = pd.DataFrame(index=prices_predict.index)

  pipeline.set_params(augmenter__timeseries_container=prices_windows)
  X = pipeline.fit_transform(X, prices_predict.Price_Target)

  price_target_model_df = X.merge(
      prices_predict[['Symbols', 'Price_Target']], 
      left_index = True, 
      right_index = True
  )

  return pipeline, price_target_model_df


def to_list(dataframe, tickers):
  """
  This function converts a complete dataframe to two lists of dataframes. The 
  first list contains the dataframes of features for each stock, and the second
  list contains the dataframes of response variables for each stock.

  INPUT:
  dataframe : The complete training dataframe
  tickers : The list of tickers to use as reference

  OUTPUT:
  df_list_X : list containing the features dataframes
  df_list_y : list containing the response variables dataframes
  """
  df_list_X = []
  df_list_y = []

  for ticker in tickers:
    X = dataframe[dataframe.Symbols == ticker]
    X.drop(['Symbols','Price_Target'], axis = 1, inplace = True)

    y = dataframe[dataframe.Symbols == ticker].Price_Target

    df_list_X.append(X)
    df_list_y.append(y)

  return df_list_X, df_list_y


def train_models(df_list_X, df_list_y):
  """
  This function trains 1 model for each stock under study and calculates it's R2

  INPUT:
  df_list_X : list containing the features dataframes
  df_list_y : list containing the response variables dataframes 

  OUTPUT:
  model_list : A list containing the trained model objects for each stock
  test_results_list : A list containing the results of applying the model on the
                      test set for each stock
  tickers_2 : A list of the tickers. This list is calculated in case the order of
              the stock was randomly changed at some point. Not necessary.
  """
  model_list = []
  test_results_list = []
  tickers_2 = []

  for df_x, df_y in zip(df_list_X, df_list_y):
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size = .25, random_state=42) 

    lm_model = GradientBoostingRegressor()

    lm_model.fit(X_train, y_train) #Fit

    #Predict and score the model
    y_test_preds = lm_model.predict(X_test)

    #Rsquared and y_test
    rsquared_score = r2_score(y_test, y_test_preds)
    length_y_test = len(y_test)

    s1 = df_y.index[0]
    ticker = s1.split('_')[0]

    print("The r-squared score for your model of stock {} was {} on {} values.".format(ticker, rsquared_score, length_y_test))

    test_results_df = pd.DataFrame(data = {'y_test' : list(y_test),
                                          'y_test_preds' : list(y_test_preds)})

    model_list.append(lm_model)
    test_results_list.append(test_results_df)
    tickers_2.append(ticker)

  return model_list, test_results_list, tickers_2


def view_performance(model_list, test_results_list, tickers_2):
  """
  This function plots the results of each model when it predicts on the test set
  against the actual values.

  OUTPUT:
  model_list : A list containing the trained model objects for each stock
  test_results_list : A list containing the results of applying the model on the
                      test set for each stock
  tickers_2 : A list of the tickers. This list is calculated in case the order of
              the stock was randomly changed at some point. Not necessary.
  """
  num_plots = len(test_results_list)

  fig, ax_array = plt.subplots(num_plots, squeeze = False,figsize=(12,20))
  for i,ax_row in enumerate(ax_array):
      for j,axes in enumerate(ax_row):
          
          coordinates = [min(test_results_list[i].y_test), max(test_results_list[i].y_test)]

          axes.set_title('{}'.format(tickers_2[i]))

          axes.set_xlabel('Price Actual Values')
          axes.set_ylabel('Price Predicted Values')

          axes.scatter(test_results_list[i].y_test, test_results_list[i].y_test_preds)
          axes.plot(coordinates, coordinates)
  plt.show()


def price_prediction(model_list, tickers_2, pipeline, target_date, t = 30):
  """
  This function calculates the prediction of the models for all stocks at a given date.
  Not necessarily the target date, it can be a date between the end_date defined
  at the beggining of the process and the target_date.

  INPUT:
  model_list : A list containing the trained model objects for each stock
  tickers_2 : A list of the tickers. This list is calculated in case the order of
              the stock was randomly changed at some point. Not necessary.
  pipeline : sklearn pipeline with the information to transform time series into 
             features for use in fitting and prediction of the machine learning model
  target_date : Date of prediction
  t : Parameter that dictates how many days are taken in each of the rolling window
      time series. Should be the same used before.

  OUTPUT:
  prices_predictions : Dataframe containing the prediction for each stock at the 
                       target_date.
  """

  today = datetime.combine(date.today(), datetime.min.time())
  end_date_f = today.strftime("%Y-%m-%d")

  start_date_f = (today - timedelta(days=2*t)).strftime("%Y-%m-%d")

  prices = download_prices(tickers_2, start_date_f, end_date_f)

  prices['id'] = prices['Symbols']
  
  prices_windows = prices
  prices_predict = pd.DataFrame(index = tickers_2)

  X_pred = pd.DataFrame(index=prices_predict.index)

  pipeline.set_params(augmenter__timeseries_container=prices_windows)
  X_pred = pipeline.transform(X_pred)

  prices_predictions = pd.DataFrame(columns = ['Date', 'Ticker', 'Prediction'])

  for model, ticker, i in zip(model_list, tickers_2, range(len(model_list))):
    des = np.array(X_pred.iloc[i])
    des = des.reshape(1, -1)
    
    prediction = model.predict(des)

    aux_df = pd.DataFrame(
      {'Date' : target_date,
      'Ticker' : ticker,
      'Prediction' : prediction
      },
      index=[0]
    )

    prices_predictions = prices_predictions.append(aux_df, ignore_index = True)
  
  return prices_predictions