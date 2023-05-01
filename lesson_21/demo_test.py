import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

df = pd.read_csv('housing.csv')

if st.button('Отобразить первые пять строк'):
    st.write(df.head())

st.write('Укажите, какой процент выборки будет выбран для обучения')
percent_train = st.slider('Размер обучающей выборки', label_visibility='collapsed', value=80, min_value=0, max_value=100)
percent_test = 100 - percent_train
if st.button('Обучить модель'):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1),
                                                        df['MEDV'],
                                                        test_size=percent_test,
                                                        random_state=2100)

    st.write('На обучение было отдано ' + str(percent_train) + ' % данных')
    st.write('На тест было отдано ' + str(percent_test) + ' % данных')
    st.write('')
    regr_model = XGBRegressor()
    regr_model.fit(X_train, y_train)
    pred = regr_model.predict(X_test)
    st.write('Обучили модель, MAE = ' + str(mean_absolute_error(y_test, pred)))