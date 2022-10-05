import streamlit as st
import pandas as pd
import mymodel as m

st.write(""" #MY FIRST APP HELLO WORLD""")
df = pd.read_csv("diabetes.csv")
st.line_chart(df)

window = st.slider("forecast diabetes")
st.write(m.run(window=window))





