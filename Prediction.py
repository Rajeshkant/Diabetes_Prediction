# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 19:00:46 2023

@author: chaithanya
"""

import numpy as np
import pickle
import streamlit as st

#loading save model
loaded_model=pickle.load(open('C:/Users/chaithanya/Downloads/trained_model.sav', 'rb'))


# Predicting the model
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'

def main():
    #giving title
    st.title('Diabetes_Prediction')
    
    #Geeting User Input Data
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Value')
    BloodPressure = st.text_input('BloodPressure')
    SkinThickness = st.text_input('skinthickness Value')
    Insulin = st.text_input('Insulin Value')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction value')
    Age = st.text_input('Age of the Person')
    
    
    #code for Prediction
    diagnosis=''
    
    #Examine for Prediction
    if st.button('Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)


if __name__=='__main__':
    main()
    

