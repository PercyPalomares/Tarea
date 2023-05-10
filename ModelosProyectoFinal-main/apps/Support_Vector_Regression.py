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
from sklearn.svm import SVR 

def app():
    st.title('Modelo - Regresión Lineal')

    start = '2004-08-18'
    end = '2022-01-27'

    st.title('Predicción del precio de las acciones')

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


    st.subheader('Seleccionamos las características para el modelo:')
    nueva_data = stock_data[['Open', 'High', 'Low','Close', 'Volume','RSI', 'MACD']]
    st.write(nueva_data.head())

    st.subheader('Dividimos los datos en entrenamiento y prueba donde el 75% de los datos para entrenamiento, 25% de datos para prueba')
    # X son nuestras variables independientes
    X = nueva_data.drop(['Close'],axis = 1)

    # y es nuestra variable dependiente
    y = nueva_data.Close

    # División 75% de datos para entrenamiento, 25% de datos para test
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)

    st.subheader('Creamos el modelo')
    svr_model = SVR(kernel='rbf')


    st.subheader('Realizamos el entrenamiento (Ajuste de parámetros) con fit()')
    svr_model.fit(X_train, y_train)

    st.subheader('Evaluamos el desempeño')
    R2 = svr_model.score(X_test,y_test)

    y_test_predict = svr_model.predict(X_test)

    RMSE = (np.sqrt(mean_squared_error(y_test, y_test_predict)))

    st.write("coeficiente de determinación R^2: ", R2)
    st.write('----------------------------------------')
    st.write('RMSE (root mean square error) nos da la diferencia entre los resultados reales y nuestros resultados calculados del modelo:')
    st.write('Rmse ',RMSE)