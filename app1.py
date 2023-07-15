#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[4]:


# Load the dataset
df = pd.read_csv("CAR DETAILS.csv")

X = df.drop('selling_price', axis=1)
y = df['selling_price']

categorical_columns = ['name', 'fuel', 'seller_type', 'transmission', 'owner']

# Apply one-hot encoding to the categorical columns
preprocessor = ColumnTransformer(
transformers=[('encoder', OneHotEncoder(), categorical_columns)],
remainder='passthrough'
)

X_encoded = preprocessor.fit_transform(X)

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor()

# Train the model
rf_regressor.fit(X_train, y_train)

import streamlit as st

# Create the Streamlit app
st.title('Car Selling Price Prediction')

# User input for feature values
user_input = {}
for column in X.columns:
    if column in categorical_columns:
        unique_values = df[column].unique()
        user_input[column] = st.selectbox(column, unique_values)
    else:
        user_input[column] = st.number_input(column, value=0)

# Transform user input to one-hot encoding
user_input_encoded = preprocessor.transform(pd.DataFrame(user_input, index=[0]))

# Make predictions using the trained model
prediction = rf_regressor.predict(user_input_encoded)

# Display the prediction
st.subheader('Prediction')
st.write(f'The predicted selling price is: {prediction[0]}')


# In[ ]: