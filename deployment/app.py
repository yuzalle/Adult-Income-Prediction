import streamlit as st
import pandas as pd
import joblib

st.header('Milestone 2 Model Deployment')
st.write("""
Created by Yuzal

Use the sidebar to select input features.
""")

@st.cache
def fetch_data():
    df = pd.read_csv('E:\\Belajar\\Hacktiv8\\PHASE1\TUGAS\\p1-ftds033-rmt-m2-yuzalle\\deployment\\new_Adult.csv')
    return df

df = fetch_data()
st.write(df)

st.sidebar.header('User Input Features')

def user_input():
    age = st.sidebar.number_input('Age', value=40)
    workclass = st.sidebar.selectbox('Workclass', df['workclass'].unique())
    education_num = st.sidebar.selectbox('Education', df['education-num'].unique())
    marital_status = st.sidebar.selectbox('Marital Status', df['marital-status'].unique())
    occupation = st.sidebar.selectbox('occupation', df['occupation'].unique())
    relationship = st.sidebar.selectbox('relationship', df['relationship'].unique())
    race = st.sidebar.selectbox('race', df['race'].unique())
    sex = st.sidebar.selectbox('sex', df['sex'].unique())
    capital_gain = st.sidebar.number_input('capital-gain', 0.0, value=58.0)
    capital_loss = st.sidebar.number_input('capital-loss', 0.0, value=58.0)
    hours_per_week = st.sidebar.number_input('hours-per-week', 0.0, value=58.0)
    native_country = st.sidebar.selectbox('native-country', df['native-country'].unique())


    data = {
        'age': age,
        'workclass': workclass,
        'education-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }
    features = pd.DataFrame(data, index=[0])
    return features


input = user_input()

st.subheader('User Input')
st.write(input)


load_model = joblib.load("my_model.pkl")


if st.button('predict'):
    prediction = load_model.predict(input)
    if prediction == 1:
        prediction = '>50k'
    else:
        prediction = '<=50k'

    st.write('Based on user input, the placement model predicted: ')
    st.write(prediction)
