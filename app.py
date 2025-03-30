import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
startTime = datetime.now()
filename = "model2.sv"
model = pickle.load(open(filename, 'rb'))
# otwieramy wcześniej wytrenowany model

chestPainType = {0: 'Atypical Angina', 1: 'Non-Anginal Pain', 2: 'Asymptomatic', 3: 'Typical Angina'}
sexD = {0: 'Male',1: 'Female'}
restingECG = {0: 'Normal', 1: 'ST', 2: 'Left Ventricular Hypertrophy'}
exerciseAngina = {0: 'Yes', 1: 'No'}
stSlope = {0: 'Up',1: 'Flat',2: 'Down'}
fastingBs = {0: 'Sugar level normal',1: 'Sugar level increased'}


# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

original_data_set = pd.read_csv('DSP_8.csv', index_col=False)

age_min = int(original_data_set['Age'].min())
age_max = int(original_data_set['Age'].max())

restingbp_min = int(original_data_set['RestingBP'].min())
restingbp_max = int(original_data_set['RestingBP'].max())

cholesterol_min = int(original_data_set['Cholesterol'].min())
cholesterol_max = int(original_data_set['Cholesterol'].max())

maxHR_min = int(original_data_set['MaxHR'].min())
maxHR_max = int(original_data_set['MaxHR'].max())

min_oldpeak = float(0)
max_oldpeak = float(original_data_set['Oldpeak'].max())


def main():

    st.set_page_config(page_title="Will he get sick?")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()



    with left:
        sex_radio = st.radio("Sex", list(sexD.keys()), format_func=lambda x: sexD[x])
        chestPainTypeRadio = st.radio("Chest pain type", list(chestPainType.keys()), index=2, format_func=lambda x: chestPainType[x])
        restingECGRadio = st.radio("Resting ecg", list(restingECG.keys()), format_func=lambda x: restingECG[x])
        exerciseAnginaRadio = st.radio("Exercise angina", list(exerciseAngina.keys()), format_func=lambda x: exerciseAngina[x])
        stSlopeRadio = st.radio("Slope", list(stSlope.keys()), format_func=lambda x: stSlope[x])
        fastingBsRadio = st.radio("FBS ", list(fastingBs.keys()), format_func=lambda x: fastingBs[x])
    with right:
        age_slider = st.slider("Age", min_value=age_min, max_value=age_max, step=1)
        restingbp_slider = st.slider("Resting bp", min_value=restingbp_min, max_value=restingbp_max, step=1)
        cholesterol_slider = st.slider("Choresterol", min_value=cholesterol_min, max_value=cholesterol_max, step=1)
        maxHR_slider = st.slider("Max HR", min_value=maxHR_min, max_value=maxHR_max, step=1)
        oldPeak_slider = st.slider("Old peak", min_value=min_oldpeak, max_value=max_oldpeak, step=0.1)

    data = [[age_slider, sex_radio, chestPainTypeRadio, restingbp_slider,
             cholesterol_slider, fastingBsRadio, restingECGRadio, maxHR_slider, exerciseAnginaRadio, oldPeak_slider, stSlopeRadio]]
    print(data)
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)


    with prediction:
        st.subheader("Will heart disease occur?")
        st.subheader(("Yes" if survival[0] == 1 else "No"))
        st.write("Prediction certainty {0:.2f} %".format(s_confidence[0][survival][0] * 100))
        if survival[0] == 1:
            st.image("https://i.kym-cdn.com/entries/icons/mobile/000/026/489/crying.jpg")
        else:
            st.image("https://cdn.pixabay.com/photo/2022/11/26/22/35/cat-7618582_960_720.jpg")

if __name__ == "__main__":
    main()
