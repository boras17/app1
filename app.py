import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename, 'rb'))
# otwieramy wcześniej wytrenowany model

pclass_d = {0:"Pierwsza",1:"Druga", 2:"Trzecia"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
sex_d = {0: 'Mężczyzna', 1: 'Kobieta'}
class_d = {0: 'Klasa 1', 1: 'Klasa 2', 2: 'Klasa 3'}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

original_data_set = pd.read_csv('DSP_1.csv', index_col=False)

age_min = int(original_data_set['Age'].min())
age_max = int(original_data_set['Age'].max())

sib_min = int(original_data_set['SibSp'].min())
sib_max = int(original_data_set['SibSp'].max())

par_min = int(original_data_set['Parch'].min())
par_max = int(original_data_set['Parch'].max())

fare_min = float(original_data_set['Fare'].min())
fare_max = float(original_data_set['Fare'].max())

def main():

    st.set_page_config(page_title="Czy przeżyje?")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    with left:
        sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        embarked_radio = st.radio( "Port zaokrętowania", list(embarked_d.keys()), index=2, format_func=lambda x: embarked_d[x])
        pclass_radio = st.radio("Klasa", list(class_d.keys()), format_func=lambda x: pclass_d[x])

    with right:
        age_slider = st.slider("Wiek", min_value=age_min, max_value=age_max, step=1)
        sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera", min_value=sib_min, max_value=sib_max, step=1)
        parch_slider = st.slider("Liczba rodziców i/lub dzieci", min_value=par_min, max_value=par_max, step=1)
        fare_slider = st.slider("Cena biletu", min_value=fare_min, max_value=fare_max, step=0.1)
    #data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]

    data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy taka osoba przeżyłaby katastrofę?")
        st.subheader(("Tak" if survival[0] == 1 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))
        if survival[0] == 0:
            st.image("https://i.kym-cdn.com/entries/icons/mobile/000/026/489/crying.jpg")
        else:
            st.image("https://cdn.pixabay.com/photo/2022/11/26/22/35/cat-7618582_960_720.jpg")


if __name__ == "__main__":
    main()