import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model

from matplotlib import style
import seaborn as sns
import math
import warnings
warnings.filterwarnings("ignore")
style.use('ggplot')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import streamlit as st
#import talib

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

def app():
    st.title('Modelo - LSTM')

    #start = '2004-08-18'
    #end = '2022-01-27'

    st.title('Predicción del precio de las acciones')

    user_input = st.text_input('Introducir cotización bursátil', 'MSFT')

    stock_data = data.DataReader(user_input, 'yahoo')
    st.subheader('Se muestra los 5 últimos registros')
    st.write(stock_data.tail())

    st.subheader('Dividimos los datos en train y test donde La LSTM se entrenará con datos de 2019 hacia atrás. La validación se hará con datos de 2020 en adelante.')
    set_entrenamiento = stock_data[:'2019'].iloc[:,1:2]
    set_validacion = stock_data['2020':].iloc[:,1:2]

    #set_entrenamiento['High'].plot(legend=True)
    #set_validacion['High'].plot(legend=True)
    #plt.legend(['Entrenamiento ( -2019)', 'Validación (2020- )'])
    #plt.show()

    st.subheader('Normalización del set de entrenamiento')
    # Normalización del set de entrenamiento
    sc = MinMaxScaler(feature_range=(0,1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

    # La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida un dato (la predicción a
    # partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
    time_step = 60
    X_train = []
    Y_train = []
    m = len(set_entrenamiento_escalado)

    for i in range(time_step,m):
        # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
        X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

        # Y: el siguiente dato
        Y_train.append(set_entrenamiento_escalado[i,0])
    
    X_train, Y_train = np.array(X_train), np.array(Y_train)

    # Reshape X_train para que se ajuste al modelo en Keras
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    st.subheader('Creamoa la Red LSTM con epochs=20')
    # Red LSTM

    dim_entrada = (X_train.shape[1],1)
    dim_salida = 1
    na = 50

    modelo = Sequential()
    modelo.add(LSTM(units=na, input_shape=dim_entrada))
    modelo.add(Dense(units=dim_salida))
    #modelo.compile(optimizer='rmsprop', loss='mse')

    modelo.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    modelo.fit(X_train,Y_train,epochs=20,batch_size=32)

    st.subheader('Realizamos la predicción del valor de las acciones')

    x_test = set_validacion.values
    x_test = sc.transform(x_test)

    X_test = []
    for i in range(time_step,len(x_test)):
        X_test.append(x_test[i-time_step:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    prediccion = modelo.predict(X_test)
    prediccion = sc.inverse_transform(prediccion)

    st.subheader('Resultado de la prediccion')
    trainScore = modelo.evaluate(X_train, Y_train, verbose=0)
    st.write('Train Score: %.2f MSE (%.2f RMSE)'% (trainScore[0], math.sqrt(trainScore[0])))


    