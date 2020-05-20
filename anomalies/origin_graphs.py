import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np

writer = pd.ExcelWriter('result.xlsx', engine='xlsxwriter')
le = preprocessing.LabelEncoder()
scaler = MinMaxScaler()

df = pd.read_csv('supermarket_sales.csv')

for col in df.columns.values:
    if df[col].dtype == 'object':
        le.fit(df[col].unique())
        df[col] = le.transform(df[col])

scaler.fit(df)

df.corr().to_excel(writer, sheet_name='corr')
df.describe().to_excel(writer, sheet_name='stats')

col_names = ['Branch', 'City', 'Customer type',  'Gender', 'Date', 'Time', 'Payment', 'cogs', 'gross margin percentage', 'gross income']
for i in range((len(col_names))):
    df.drop(col_names[i], axis=1, inplace=True)

df.to_excel(writer, sheet_name = 'need_columns')
writer.save()

x_column_name = 'Invoice ID'
x_values = df[x_column_name]
df.drop(x_column_name, axis=1, inplace=True)
col_names = df.columns.values
N = len(df.columns.values)

fig, axs = plt.subplots(6, 1)
axs = axs.ravel()
axs[0].scatter(x_values, df['Product line'], s=1)
axs[0].set_xlabel(x_column_name, fontsize=6)
axs[0].set_ylabel('Product line', fontsize=6)
axs[0].tick_params(labelsize=6)

axs[1].scatter(x_values, df['Unit price'], s=1)
axs[1].set_xlabel(x_column_name, fontsize=6)
axs[1].set_ylabel('Unit price', fontsize=6)
axs[1].tick_params(labelsize=6)

axs[2].scatter(x_values, df['Quantity'], s=1)
axs[2].set_xlabel(x_column_name, fontsize=6)
axs[2].set_ylabel('Quantity', fontsize=6)
axs[2].tick_params(labelsize=6)

axs[3].scatter(x_values, df['Tax 5%'], s=1)
axs[3].set_xlabel(x_column_name, fontsize=6)
axs[3].set_ylabel('Tax 5%', fontsize=6)
axs[3].tick_params(labelsize=6)

axs[4].scatter(x_values, df['Total'], s=1)
axs[4].set_xlabel(x_column_name, fontsize=6)
axs[4].set_ylabel('Total', fontsize=6)
axs[4].tick_params(labelsize=6)

axs[5].scatter(x_values, df['Rating'], s=1)
axs[5].set_xlabel(x_column_name, fontsize=6)
axs[5].set_ylabel('Rating', fontsize=6)
axs[5].tick_params(labelsize=6)

plt.show()
