import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import joblib

col_names = ['Refrigerator',
    'Clothes washer',
    'Clothes Iron',
    'Computer',
    'Oven',
    'Play',
    'TV',
    'Sound system']




def classifyConsumption(df):

    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Dia'] = df['Fecha'].dt.day
    df['Hora'] = df['Fecha'].dt.hour
    df['Minuto'] = df['Fecha'].dt.minute

    del df['Unnamed: 0']
    #del df['Fecha']

    columnas = df.columns.tolist()
    ultimas_tres_columnas = columnas[-3:]
    nuevas_columnas = [columnas[0]] + ultimas_tres_columnas + columnas[1:-3]
    df = df[nuevas_columnas]

    df.columns = df.columns.str.replace(' ', '')
    df = df.rename(columns={'Medidor[W]': 'Medidor'})

    

    modc0 = joblib.load('modelo_clasificacion_d0.pkl')
    modc1 = joblib.load('modelo_clasificacion_d1.pkl')
    modc2 = joblib.load('modelo_clasificacion_d2.pkl')
    modc3 = joblib.load('modelo_clasificacion_d3.pkl')
    modc4 = joblib.load('modelo_clasificacion_d4.pkl')
    modc5 = joblib.load('modelo_clasificacion_d5.pkl')
    modc6 = joblib.load('modelo_clasificacion_d6.pkl')
    modc7 = joblib.load('modelo_clasificacion_d7.pkl')




    modelosC = {
        "modc0" : modc0,
        "modc1" : modc1,
        "modc2" : modc2,
        "modc3" : modc3,
        "modc4" : modc4,
        "modc5" : modc5,
        "modc6" : modc6,
        "modc7" : modc7,
    }



    dfResultC = df.copy()

    X = df.iloc[:,1:]
    

    def prediccionesC(X):
        for i in range(8):
            y_pred = modelosC[f"modc{i}"].predict(X)
            dfResultC[col_names[i]] = y_pred


    prediccionesC(X)

    del dfResultC["Dia"]
    del dfResultC["Hora"]
    del dfResultC["Minuto"]


    
    return dfResultC


def predictConsumption(df):

    
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Dia'] = df['Fecha'].dt.day
    df['Hora'] = df['Fecha'].dt.hour
    df['Minuto'] = df['Fecha'].dt.minute

    del df['Unnamed: 0']
    #del df['Fecha']

    columnas = df.columns.tolist()
    ultimas_tres_columnas = columnas[-3:]
    nuevas_columnas = [columnas[0]] + ultimas_tres_columnas + columnas[1:-3]
    df = df[nuevas_columnas]

    df.columns = df.columns.str.replace(' ', '')
    df = df.rename(columns={'Medidor[W]': 'Medidor'})

    
    dfResultR = df.copy()

    X = df.iloc[:,1:]


    
    modr0 = joblib.load('modelo_regDTN_d0.pkl')
    modr1 = joblib.load('modelo_regDTN_d1.pkl')
    modr2 = joblib.load('modelo_regDTN_d2.pkl')
    modr3 = joblib.load('modelo_regDTN_d3.pkl')
    modr4 = joblib.load('modelo_regDTN_d4.pkl')
    modr5 = joblib.load('modelo_regDTN_d5.pkl')
    modr6 = joblib.load('modelo_regDTN_d6.pkl')
    modr7 = joblib.load('modelo_regDTN_d7.pkl')

    modelosR = {
        "modc0" : modr0,
        "modc1" : modr1,
        "modc2" : modr2,
        "modc3" : modr3,
        "modc4" : modr4,
        "modc5" : modr5,
        "modc6" : modr6,
        "modc7" : modr7,
    }

    def prediccionesR(X):
        for i in range(8):
            y_pred = modelosR[f"modc{i}"].predict(X)
            dfResultR[col_names[i]] = y_pred

    prediccionesR(X)

    del dfResultR["Dia"]
    del dfResultR["Hora"]
    del dfResultR["Minuto"]


    return dfResultR

