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

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def app():
    st.title('Modelo - SVM de Clasificación')

    start = '2004-08-18'
    end = '2022-01-27'

    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil', 'MSFT')

    stock_data = data.DataReader(user_input, 'yahoo', start, end)
    st.subheader('Datos del 2004 al 2022 (se muestra los 5 últimos registros)')
    st.write(stock_data.tail())

    st.subheader('Variables adicionales:')
    
    st.subheader('RSI (Indicador de fuerza relativa):')

    delta = stock_data['Close'].diff()
    up = delta.clip(lower = 0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up/ema_down
    stock_data['RSI'] = 100 - (100/(1+rs))
    st.write(stock_data)

    st.subheader('MACD (Media Móvil de Convergencia/Divergenci):')

    exp1 = stock_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = exp1 -exp2
    stock_data['Signal line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

    st.write(stock_data.head())

    #Quitamos las filas que tengan el valor de NaN
    stock_data=stock_data.dropna()
    #stock_data

    st.subheader('Definir las variables Explicativas')
    # Crear variables predictoras
    stock_data['Open-Close'] = stock_data.Open - stock_data.Close
    stock_data['High-Low'] = stock_data.High - stock_data.Low

    # Almacenar todas las variables predictoras en una variable X
    X = stock_data[['Open-Close', 'High-Low', 'RSI', 'MACD']]
    st.write(X)

    st.subheader('Definir la variable Objetivo')
    # Variable objetivo
    #Comparamos el cierre anterior con el actual
    y = np.where(stock_data['Close'].shift(-1) > stock_data['Close'], 1, 0)
    st.write(y)

    st.subheader('dividir los datos en train y test')
    split_percentage = 0.8
    split = int(split_percentage*len(stock_data))

    # Train dataset
    X_train = X[:split]
    y_train = y[:split]

    # Test dataset
    X_test = X[split:]
    y_test = y[split:]

    st.subheader('Usamos la función SVC() de la biblioteca sklearn.svm')
    # Support vector classifier
    cls = SVC().fit(X_train, y_train)

    st.subheader('Precisión del clasificador')
    st.write('********************************************')
    st.write('Para el conjunto de datos de Entrenamiento: ')
    y_pred = cls.predict(X_train)
    st.write("Train Accuracy: ", accuracy_score(y_train, y_pred))

    st.write('********************************************')
    st.write('Para el conjunto de datos de Pueba: ')
    y_pred = cls.predict(X_test)
    st.write("Test Accuracy: ", accuracy_score(y_test, y_pred))







   