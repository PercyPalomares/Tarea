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
from sklearn.ensemble import RandomForestClassifier

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report


def app():
    st.title('Modelo - Random Forest Classifie')

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

    st.subheader('Creamos la variable objetivo, Tendencia:')
    stock_data['Tendencia'] = np.where(stock_data['Close'].shift(-1) > stock_data['Close'], 1, 0)
    st.write(stock_data)

    st.subheader('Seleccionamos las características para el modelo:')
    nueva_data = stock_data[['Open', 'High', 'Low','Close', 'Volume','RSI', 'MACD', 'Tendencia']]
    st.write(nueva_data.head())

    st.subheader('Dividimos los datos en entrenamiento y prueba donde el 75% de los datos para entrenamiento, 25% de datos para prueba')
    # X son nuestras variables independientes
    X = nueva_data.drop(['Tendencia'],axis = 1)

    # y es nuestra variable dependiente
    y = nueva_data.Tendencia

    # División 75% de datos para entrenamiento, 25% de datos para test
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)

    st.subheader('Creamos el modelo de Bosques Aleatorios y configuramos el número de estimadores (árboles de decisión)')
    st.subheader('usamos RandomForestClassifier')
    BA_model = RandomForestClassifier(n_estimators = 19, random_state = 2016,min_samples_leaf = 8,)

    st.subheader('Entrenamos el modelo con fit()')
    BA_model.fit(X_train, y_train)

    st.subheader('Calculamos el Accuracy promedio (Usando datos de Test)')
    st.write("Precisión del modelo Random Forest Classifier: {:>7.4f}".format(BA_model.score(X_test, y_test)))

    st.subheader('Matriz de Confusión')
    # Predicción del modelo, usando los datos de prueba
    y_pred = BA_model.predict(X_test)
    matriz = confusion_matrix(y_test,y_pred)

    plot_confusion_matrix(conf_mat=matriz, figsize=(2,2), show_normed=False)
    #plt.tight_layout()
    st.pyplot()
