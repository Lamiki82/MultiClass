import streamlit as st
import joblib
import pandas as pd

model_pipe = joblib.load('penguinspipe.pkl')
print('modello caricato')

island = st.selectbox("Isola?",["Torgersen","Dream","Biscoe"])
bill_length_mm = st.number_input("Lunghezza becco?", 0.00, 80.00, 40.1)
bill_depth_mm = st.number_input("Larghezza becco?", 0.00, 50.00, 19.1)
flipper_length_mm = st.number_input("Lunghezza pinne?", 10, 300, 180)
body_mass_g = st.number_input("Peso?", 500, 10000, 3750)
sex = st.selectbox("Sesso?",["male","female"])
                             
# island= 'Torgersen'
# bill_length_mm = 39.1
# bill_depth_mm = 18.7
# flipper_length_mm = 181
# body_mass_g = 3750
# sex = 'male'
#cntr K poi control c oppure u per togliere

data = {
        "island": [island],
        "bill_length_mm": [bill_length_mm],
        "bill_depth_mm": [bill_depth_mm],
        "flipper_length_mm":[flipper_length_mm],
        "body_mass_g": [body_mass_g],
        "sex": [sex],
        }

input_df = pd.DataFrame(data)
res = model_pipe.predict(input_df).astype(int)[0]
print(res)

classes = {0:'Adelie',
           1:'Gentoo',
           2:'Chinstrap',
           }

y_pred = classes[res]

if st.button("Predicts"):
    st.success(f"La specie del pinguino è {y_pred}")
