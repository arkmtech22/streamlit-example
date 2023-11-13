#import statements
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

#App description
st.markdown('''
# Diabetes Detector 
This app detects if you have diabetes based on Machine Learning!
- App built by 
- Dataset resource: Pima Indian Datset (United States National Institutes of Health). 
- Note: User inputs are taken from the sidebar. It is located at the top left of the page (arrow symbol). The values of the parameters can be changed from the sidebar.  
''')
st.write('---')

df = pd.read_csv(r'diabetes.csv')

#titles
st.sidebar.header('Patient Data')
st.subheader('Training Dataset')
st.write(df.describe())


#train data. Fun!
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


#User reports
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

#Visualizations, this is where the beauty begins.
st.title('Graphical Patient Report')

if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'

#Good old glucose
st.header('Glucose Value Graph (Yours vs Others)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='Purples')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['glucose'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,250,20))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)


#Insulin
st.header('Insulin Value Graph (Yours vs Others)')
fig_insulin = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rainbow')
ax10 = sns.scatterplot(x = user_data['age'], y = user_data['insulin'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_insulin)


#Famous saying BP
st.header('Blood Pressure Value Graph (Yours vs Others)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Blues')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['bp'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,320,20))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)


#Did'nt even know this before nutrition training 
st.header('BMI Value Graph (Yours vs Others)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='Greens')
ax12 = sns.scatterplot(x = user_data['age'], y = user_data['bmi'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,75,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)


#Something new, cool
st.header('DPF Value Graph (Yours vs Others)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='rocket')
ax14 = sns.scatterplot(x = user_data['age'], y = user_data['dpf'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,3.2,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)


#Don't even know how thats related to diabetes.The dataset was females only though
st.header('Pregnancy count Graph (Yours vs Others)')
fig_pregn = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'magma')
ax2 = sns.scatterplot(x = user_data['age'], y = user_data['pregnancies'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_pregn)


#Wonder how people measure that 
st.header('Skin Thickness Value Graph (Yours vs Others)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Reds')
ax8 = sns.scatterplot(x = user_data['age'], y = user_data['skinthickness'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)


#Finally!
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'Congratulations, you are not Diabetic'
else:
  output = 'Unfortunately, you are Diabetic'
st.title(output)
