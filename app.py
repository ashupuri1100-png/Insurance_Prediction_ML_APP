#step 1 : import important modules 
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import streamlit as st

# this streamlit is for web based application project 

#web Page code
st.title("HEALTH INSURANCE PREDICTION")

img_url = "https://www.magnific.com/free-photos-vectors/health-insurance"
st.image(img_url)


# LOAD DATA and ML MODEL PART

#STEP 2: Load the dataset
url = "https://imgs.search.brave.com/SbQSqTHphByYJybBtaTMr9MAFhLvlDMOPfkHBe2rgk8/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pbWcu/bWFnbmlmaWMuY29t/L2ZyZWUtdmVjdG9y/L2Z1bi1oZWFsdGgt/aW5zdXJhbmNlLWNv/bXBvc2l0aW9uXzIz/LTIxNDc2NjE5Mjcu/anBnP3NlbXQ9YWlz/X3Rlc3RfYiZ3PTc0/MCZxPTgw"
df = pd.read_csv(url)
df.sample()


#step 3 : EDA: Exploratory Data Analysis
df.drop("Customer_ID",axis = 1,inplace = True)

df['Previous_Insurance'] = df['Previous_Insurance'].map({'No':0,'Yes':1})
df['Insurance_Bought'] = df['Insurance_Bought'].map({'No':0,'Yes':1})


#step 4: Divide dataset into features and target
X=df.iloc[:,:-1]
y = df.iloc[:,-1]


#step 5: Divide data into Training & Testing part
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.3)


#step 6 : Train Model
model = LogisticRegression()
model.fit(X_train,y_train)


# Show data sample 
st.write(df.head())


#create side bar for user input form
st.sidebar.title("Fill Customer Details")
st.sidebar.image(img_url)

all_ans = []

for index,col_name in enumerate(X.columns):
    min_v = X[col_name].min()
    max_v = X[col_name].max()

    if col_name != "Previous_Insurance":
        value = st.sidebar.slider(
            f"select value for {col_name}",
            min_value = int(min_v),
            max_value = int(max_v)
        )
    else:
        value = st.sidebar.number_input(
            f"Select value for {col_name} (0: No, 1: Yes)",
            min_value = 0,
            max_value = 1,
            step = 1
        )

    all_ans.append(value)

ud = {j:all_ans[i] for i,j in enumerate(X.columns)}
user_df = pd.DataFrame([all_ans], columns = X.columns)
st.write(user_df)


#====================================================================prediction==============================================================================================================================#

if st.button("CLick to Predict"):
    with st.spinner("predicting.."):
        import time
        time.sleep(2)

    final_ans = model.predict([all_ans])[0]

    if final_ans == 0:
        st.info("✖️customer will not buy the insurance✖️")
    else:
        st.success("☑️customer will buy the insurance☑️")
