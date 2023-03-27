#diabetes detector
#First ML App deployemnt on Streamlit
#Works perfectly!
#85+ accuracy, 

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

# #Most important for users
# st.subheader('Lets raise awareness for diabetes and show our support for diabetes awareness and help many patients around the world.')
# st.write("World Diabetes Day: 14 November")

# st.sidebar.subheader("""An article first-web-app.html""")
# st.write("Dataset citation .")
# st.write("Original owners of the dataset: Original owners: National Institute of Diabetes and Digestive and Kidney Diseases (b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu) Research Center, RMI Group Leader Applied Physics Laboratory The Johns Hopkins University Johns Hopkins Road Laurel, MD 20707 (301) 953-6231 © Date received: 9 May 1990")
# st.write("This dataset is also available on the UC Irvine Machine Learning Repository")
# st.write("Dataset License: Open Data Commons Public Domain Dedication and License (PDDL)")

# st.write("Disclaimer: This is just a learning project based on one particular dataset so please do not depend on it to actually know if you have diabetes or not. It might still be a false positive or false negative. A doctor is still the best fit for the determination of such diseases.")
# image = Image.open('killocity (3).png')

# st.image(image, use_column_width=True)

# st.sidebar.header('Diabetes Prediction')
# select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
# # select = st.sidebar.selectbox('Select Form', ['Decision Tree'], key='1')
# if not st.sidebar.checkbox("Hide", True, key='1'):
#     st.title('Diabetes Prediction(Only for females above 21years of    Age)')
#     name = st.text_input("Name:")
#     pregnancy = st.number_input("No. of times pregnant:")
#     glucose = st.number_input("Plasma Glucose Concentration :")
#     bp =  st.number_input("Diastolic blood pressure (mm Hg):")
#     skin = st.number_input("Triceps skin fold thickness (mm):")
#     insulin = st.number_input("2-Hour serum insulin (mu U/ml):")
#     bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
#     dpf = st.number_input("Diabetes Pedigree Function:")
#     age = st.number_input("Age:")
# submit = st.button('Predict')
# if submit:
#         prediction = classifier.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
#         if prediction == 0:
#             st.write('Congratulation',name,'You are not diabetic')
#         else:
#             st.write(name," we are really sorry to say but it seems like you are Diabetic.")
            
            
 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
pip install scikit-fuzzy
import numpy as np
import skfuzzy as fuzz
import skfuzzy as fuzz

# Load the diabetes dataset
# diabetes_df = pd.read_csv("/content/drive/My Drive/COLAB/diabetes.csv")
diabetes_df = pd.read_csv(r'diabetes.csv')

# Split the dataset into features and labels
X = diabetes_df.iloc[:, :-1].values
y = diabetes_df.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling using the StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Binary classification using logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
acc_logreg = accuracy_score(y_test, y_pred_logreg)
cr_logreg = classification_report(y_test, y_pred_logreg)
cm_logreg = confusion_matrix(y_test, y_pred_logreg)

# Multiclass classification using decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
cr_dt = classification_report(y_test, y_pred_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)

# Rule-based classification using a fuzzy expert system
def get_fuzzy_expert_system(X_train, y_train, X_test):
    # Divide the output space into three fuzzy sets
    diabetes = fuzz.trimf([0, 0, 1], [0, 0.5, 1])
    prediabetes = fuzz.trimf([0, 1, 2], [0.5, 1, 1.5])
    normal = fuzz.trimf([1, 1, 2], [1, 1.5, 2])

    # Define the fuzzy rules
    rule1 = np.fmax(diabetes, normal)
    rule2 = prediabetes
    rule3 = np.fmax(prediabetes, diabetes)

    # Combine the rules using the OR operator
    or_rule = np.fmax(rule1, np.fmax(rule2, rule3))

    # Compute the output using the weighted average defuzzification method
    y_train_fuzzy = np.column_stack([fuzz.interp_membership(X_train[:, i], X_train[:, i], or_rule[i]) for i in range(len(X_train[0]))])
    y_test_fuzzy = np.column_stack([fuzz.interp_membership(X_test[:, i], X_test[:, i], or_rule[i]) for i in range(len(X_test[0]))])
    y_pred_fuzzy = np.argmax(y_train_fuzzy, axis=1)
    y_pred_fuzzy_test = np.argmax(y_test_fuzzy, axis=1)

    return y_pred_fuzzy_test

y_pred_fuzzy = get_fuzzy_expert_system(X_train, y_train, X_test)
acc_fuzzy = accuracy_score(y_test, y_pred_fuzzy)
cr_fuzzy = classification_report(y_test, y_pred_fuzzy)
cm_fuzzy = confusion_matrix(y_test, y_pred_fuzzy)

# Fuzzy logic-based classification using fuzzy c-means clustering
def get_fuzzy_cmeans(X_train, y_train, X_test):
    # Use the elbow method to determine the number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X_train)
        wcss.append(kmeans.inertia_)
    st.line_chart(wcss)
    k = st.slider('Select number of clusters for fuzzy c-means:', min_value=1, max_value=10)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_train.T, k, 2, error=0.005, maxiter=1000, init=None)
    u_pred = fuzz.cluster.cmeans_predict(X_test.T, cntr, 2, error=0.005, maxiter=1000)
    y_pred_fcm = np.argmax(u_pred, axis=0)
    return y_pred_fcm

y_pred_fcm = get_fuzzy_cmeans(X_train, y_train, X_test)
y_pred_fuzzy_test = np.argmax(y_pred_fcm, axis=0)
acc_fcm = accuracy_score(y_test, y_pred_fuzzy_test)
cr_fcm = classification_report(y_test, y_pred_fuzzy_test)
cm_fcm = confusion_matrix(y_test, y_pred_fuzzy_test)

# Display the results
st.write('## Diabetes Prediction with Supervised Machine Learning')
st.write('### Binary Classification using Logistic Regression')
st.write('Accuracy:', acc_logreg)
st.write('Classification Report:\n', cr_logreg)
st.write('Confusion Matrix:\n', cm_logreg)
st.write('### Multiclass Classification using Decision Tree')
st.write('Accuracy:', acc_dt)
st.write('Classification Report:\n', cr_dt)
st.write('Confusion Matrix:\n', cm_dt)
st.write('### Rule-based Classification using a Fuzzy Expert System')
st.write('Accuracy:', acc_fuzzy)
st.write('Classification Report:\n', cr_fuzzy)
st.write('Confusion Matrix:\n', cm_fuzzy)
st.write('### Fuzzy Logic-based Classification using Fuzzy C-means Clustering')
st.write('Accuracy:', acc_fcm)
st.write('Classification Report:\n', cr_fcm)
st.write('Confusion Matrix:\n', cm_fcm)
            

