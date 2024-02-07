# -*- coding: utf-8 -*-
"""
CYBER362 Lab 2 Section B
@author: lfg5289
"""

# Imports
import pandas as pd
import numpy as np

# Brain Size
brain_data = pd.read_csv('brain_size.csv', sep=';', na_values=".")

# What is the mean value for FSIQ for the full population?
print("FSIQ mean for population: " + str(brain_data['FSIQ'].mean()))

# How many males and females were included in this study (total)?
print("Females: " + str(len(brain_data[brain_data['Gender'] == 'Female'])))
print("Males: " + str(len(brain_data[brain_data['Gender'] == 'Male'])))

# Create a Python statement after line 51 that calculates the average (mean) value of MRI counts for males and females?
print("Female MRI Count mean: " + str(brain_data[brain_data['Gender'] == 'Female']['MRI_Count'].mean()))
print("Male MRI Count mean: " + str(brain_data[brain_data['Gender'] == 'Male']['MRI_Count'].mean()))

# Payment Fraud
fraud_data = pd.read_csv('payment_fraud.csv')

# Copy and paste output from the following lines:
print(fraud_data.describe())
print(fraud_data.shape)
print(fraud_data.loc[0:3])
print(fraud_data.iloc[5:8, 0:3])
print(fraud_data.iloc[[1, 5, 8], [1, 4]])
print(fraud_data.mean(numeric_only = True))
ts = fraud_data.apply(np.cumsum)
print(ts.plot)

# Provide code that returns all samples whose payment method was conducted using ‘paypal’
print(fraud_data[fraud_data['paymentMethod'] == 'paypal'])

# Provide code for obtaining fraudulent transactions samples whose payment method was conducted using ‘paypal’
print(fraud_data[(fraud_data['paymentMethod'] == 'paypal') & (fraud_data['label'] == 1)])