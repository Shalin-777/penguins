import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Penguine Species Prediction ML App")
st.info("This is end-to-end Machine Learning App")

with st.expander("Data"):
  st.write("**Raw Data**")
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
  df

st.write("Input Variables")
X_raw = df.drop("species",axis = 1)
X_raw

st.write("Target Variables")
y_raw = df.species
y_raw

st.write("Descriptive Statistics")
des = df. describe()
des

st.write("More information about Data")
inf = df.info()
print (inf)

with st.expander("Data Visualization"):
    st.scatter_chart(data=df, x="bill_length_mm", y="body_mass_g", color="species")
    
    st.write("Box Plot: Body Mass by Species")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="species", y="body_mass_g", ax=ax)
    st.pyplot(fig)



with st.expander("Input data"):
  pass

with st.expander("Data Preperation"):
  pass

with st.sidebar:
    st.header("Input Variables")
    island = st.selectbox("Island", ("Biscoe","Dream", "Torgersen"))
    bill_length_mm = st.slider("Bill length (mm)", 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider("Bill depth (mm)", 13.1, 21.5, 17.2)
    flipper_depth_mm = st.slider("Flipper depth (mm)", 172.0, 231.0, 201.0)
    body_mass_g = st.slider("Body mass (g)", 2700.0, 6300.0, 4207.0)
    gender = st.selectbox("Gender", ("male", "female"))

    data = {
        "island": island,
        "bill_length_mm": bill_length_mm,
        "bill_length_mm": bill_length_mm,
        "flipper_depth_mm": flipper_depth_mm,
        "body_mass_g": body_mass_g,
        "gender": gender
    }

    input_df = pd.DataFrame(data,index=[0])
    input_penguins = pd.concat([input_df, X_raw], axis=0)

    with st.expander("Input data"):
        st.write("**Input data**")
        input_df

        st.write("**Combined data**")
        input_penguins

      # One hot encoding for X
    encode = ["island","sex"]
    df_penguins = pd.get_dummies(input_penguins, prefix = encode)
    X = df_penguins[1:]
    input_row = df_penguins[:1]
     
      # One hot encoding for y
    target_mapper = {
        "Adelie": 0,
        "Chinstrap": 1,
        "Gentoo" : 2
      }
      def target_encode(val):
        return target_mapper[val]

      y = y_raw.apply(target_encode)
