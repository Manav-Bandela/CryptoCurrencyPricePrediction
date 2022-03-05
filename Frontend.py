import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


st.title('Cryptocurrency Price Predictor')

tickers = ('BTC-USD', 'ETH-USD', 'XRP-USD', 'DOGE-USD', 'SOL-USD', 'SHIB-USD', 'ADA-USD', 'USDT-USD', 'WRX-USD', 'DOT-USD')

dropdown = st.multiselect('Pick Your Crypto', tickers)


start = st.date_input('Start', value = pd.to_datetime('2013-01-01'))
end = st.date_input('End', value = pd.to_datetime('today'))


def pred(df):
    training_df = df.drop(['Adj Close'], axis = 1)
    #training_df.head()
    
    scaler = MinMaxScaler()
    training_df = scaler.fit_transform(training_df)
    #training_df
    
    X_train = []
    Y_train = []
    training_df.shape[0]
    
    for i in range(60, training_df.shape[0]):
        X_train.append(training_df[i-60:i])
        Y_train.append(training_df[i,0])
    
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_train.shape
    
    
    regressor = Sequential()
    regressor.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
    regressor.add(Dropout(0.3))
    
    regressor.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
    regressor.add(Dropout(0.4))
    
    regressor.add(LSTM(units = 120, activation = 'relu'))
    regressor.add(Dropout(0.5))
    
    regressor.add(Dense(units =1))
    
    
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    regressor.fit(X_train, Y_train, epochs = 20, batch_size =50)
    
    df_test = df
    past_60_days = df.tail(60)
    df= past_60_days.append(df_test, ignore_index = True)
    df = df.drop(['Adj Close'], axis = 1)
    df.head()
    
    inputs = scaler.transform(df)
    #inputs
    
    X_test = []
    Y_test = []
    for i in range (60, inputs.shape[0]):
        X_test.append(inputs[i-60:i])
        Y_test.append(inputs[i, 0])

    X_test, Y_test = np.array(X_test), np.array(Y_test)
    X_test.shape, Y_test.shape
    

    Y_pred = regressor.predict(X_test)
    #Y_pred, Y_test
    
    scaler.scale_
    
    scale = 1/5.18164146e-05
    
    Y_test = Y_test*scale
    Y_pred = Y_pred*scale
    #Y_pred
    #Y_test
    
    
    
    plt.figure(figsize=(14,5))
    plt.title('Cryptocurrency Price Prediction using RNN-LSTM')
    plt.plot(Y_test, color = 'red', label = 'Real Price')
    plt.plot(Y_pred, color = 'green', label = 'Predicted Price')
    plt.legend()
    a = plt.gca()
    a.axes.get_xaxis().set_visible(False)
    x= plt.savefig('graph.png')
    st.pyplot(x) 

    
if len(dropdown) > 0 and len(dropdown) < 2:
    df = pred(yf.download(dropdown,start,end))
    df_1 = yf.download(dropdown,start,end)
    vol = df_1['Volume']
    st.line_chart(vol)

if len(dropdown) == 2:
    df_2 = yf.download(dropdown,start,end)
    df_3 = yf.download(dropdown,start,end)
    vol_1 = df_2['Volume']
    vol_2 = df_3['Volume']
    st.line_chart(vol_1)
    #st.line_chart(vol_2)