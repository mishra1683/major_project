import streamlit as st
import video as v
import numpy as np
import pandas as pd
import song as s
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
import pickle

st.title("Stress Checker")
st.write("Let's firstly see your face!!")


p=v.func()
if len(p)/50 > 0.3:
    st.write("It's looks like you are in stress. For further clarification we would like to do another check up")
    loaded_model = pickle.load(open(r"tests\model\module_2.sav", 'rb'))
    with st.form(key='form1'):
        a=st.number_input("Age")
        b=st.number_input("Gender")
        c=st.number_input("Diabetes")
        d=st.number_input("Heart Rate")
        e=st.number_input(label="systolic blood pressure",step=1.,format="%.2f")
        f=st.number_input("Diastolic blood pressure")
        g=st.number_input("Temperature")
        h=st.form_submit_button(label='Submit')
        if h:
            with st.expander("Results"):
                r=list(loaded_model.predict(pd.DataFrame(columns=['age','gendera','diabetes','heart rate','Systolic blood pressure','Diastolic blood pressure','temperature'],data=np.array([a,b,c,d,e,f,g]).reshape(1,7))))
                m=r[0].item()
                if m==0:
                    st.write('You are Not Stressed.')
                    
                else:
                    st.write("You are Stressed, listen to these Songs.")
                    st.write(s.abc())
else:
    st.write("You are Not Stressed.")
    
    

