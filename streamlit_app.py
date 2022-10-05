import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from PIL import Image

df = pd.read_csv(r'diabetes.csv')
st.sidebar.header('Patient Data')
st.subheader('Training Dataset')
st.write(df.describe())

x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

def user_report():
  glucose = st.sidebar.slider('Glucose', 0,250, 120 )
  insulin = st.sidebar.slider('Insulin', 0,850, 90 )
  bp = st.sidebar.slider('Blood Pressure', 0,300, 85 )
  bmi = st.sidebar.slider('BMI', 0,70, 22 )
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,3.0, 0.8 )
  age = st.sidebar.slider('Age', 21,120, 55 )
  pregnancies = st.sidebar.slider('Pregnancies', 0,10, 1 )
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 35 )

  user_report_data = {
      'glucose':glucose,
      'insulin':insulin,
      'bp':bp,
      'bmi':bmi,
      'dpf':dpf,
      'age':age,
      'pregnancies':pregnancies,
      'skinthickness':skinthickness,
         
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data



user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

st.title('Graphical Patient Report')



if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'


st.header('Glucose Value Graph (Yours vs Others)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='Purples')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['glucose'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,250,20))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)

st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'Congratulations, you are not Diabetic'
else:
  output = 'Unfortunately, you are Diabetic'
st.title(output)



