# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:15:03 2023

@author: ravul
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("D:/New folder/pythonProject/Diabetes_pickle.sav",'rb'))

def diabetes_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if(prediction[0]==0):
        return "Not diabetic"
    else:
        return "Diabetic"
            
def main():
    
    st.title("Get fit and healthy")
    pregnencies = st.text_input('no.of pregnancies')
    Glucose = st.text_input('glucose level')
    BloodPressure = st.text_input('bp value')
    SkinThickness = st.text_input('skin thickness value')
    Insulin  = st.text_input('insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction')
    Age = st.text_input('age')
    
    diagnosis = ''
    if st.button('Diagnosis'):
        diagnosis = diabetes_prediction([pregnencies,Glucose, BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)
    
    
if _name_ =='_main_':
    main()

    
    
    
    