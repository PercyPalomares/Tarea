import streamlit as st
from multiapp import MultiApp
from apps import home, Random_Forest_Classifier, Regresión_Lineal, SVM_de_Clasificación, Logistic_Regression, Support_Vector_Regression, LSTM # import your app modules here

app = MultiApp()

st.markdown("""
#  Inteligencia de Negocios - Grupo B

""")

# Add all your application here
app.add_app("Home", home.app)
#app.add_app("Modelo", model.app)
app.add_app("Random Forest Classifier", Random_Forest_Classifier.app)
app.add_app("Regresión Lineal", Regresión_Lineal.app)
app.add_app("SVM de Clasificación", SVM_de_Clasificación.app)
app.add_app("Logistic Regression", Logistic_Regression.app)
app.add_app("Support Vector Regression", Support_Vector_Regression.app)
app.add_app("LSTM", LSTM.app)
# The main app
app.run()