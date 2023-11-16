import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st 
# import tensorflow.compat.v2 as tf
import yfinance as yf

start='2010-01-01'
end='2019-12-31'
st.title('Stock Trend Prediction')
user_input=st.text_input('Enter the stock ticker', 'AAPL')

df=yf.download(user_input, start=start,end=end)
st.subheader(f'Data of {user_input}')
st.write(df.describe())

st.subheader('Closing Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()

st.subheader('Moving Average 100')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
st.pyplot(fig)

st.subheader('Moving Average 200')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma200,'g')
st.pyplot(fig)


data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array= scaler.fit_transform(data_training)

model =load_model('keras_model.h5')
past_100_days=data_training.tail(10)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(data_training_array[i-100:i])
    y_test.append(data_training_array[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted=model.predict(x_test)
scaler=scaler.scale_
scale_factor=1/0.02099517
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

st.subheader('Prediction')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original')
plt.plot(y_predicted,'r',label='Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



# df=df.reset_index();
# df=df.drop(['Date','Adj Close'], axis=1)



# st.subheader('Data from 2010-2019')
# st.write(df.describe())
