import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df_origin = pd.read_excel(open('result.xlsx', 'rb'), sheet_name='Dataset')
df_em = pd.read_excel(open('result.xlsx', 'rb'), sheet_name='EM')
df_c = pd.read_excel(open('result.xlsx', 'rb'), sheet_name='C')

col_names_drop = ['Branch', 'City', 'Customer type',  'Gender', 'Date', 'Time', 'Payment', 'gross margin percentage','Invoice ID', 'Product line']
for i in range((len(col_names_drop))):
    df_em.drop(col_names_drop[i], axis=1, inplace=True)
    df_origin.drop(col_names_drop[i], axis=1, inplace=True)
    df_c.drop(col_names_drop[i], axis=1, inplace=True)

df_origin_miss = df_origin[df_c == False]
df_em_miss = df_em[df_c == False]
origin_miss = []
em_miss = []

for j in range(8):
    temp_orig = []
    temp_em = []
    for i in range(1000):
        if np.isnan(df_origin_miss.iloc[i, j]) == False:
            temp_orig.append(df_origin_miss.iloc[i, j])
        if np.isnan(df_em_miss.iloc[i, j]) == False:
            temp_em.append(df_em_miss.iloc[i, j])
    origin_miss.append(temp_orig)
    em_miss.append(temp_em)

for i in range(len(origin_miss)):
    print(len(origin_miss[i]))

fig, axs = plt.subplots(5, 1)
axs = axs.ravel()

axs[0].plot(np.arange(0, len(origin_miss[1]), 1), origin_miss[1], c='blue', label='origin')
axs[0].plot(np.arange(0, len(em_miss[1]), 1), em_miss[1], c='magenta', label='EM-algorithm')
axs[0].set_xlabel('Number', fontsize=6)
axs[0].set_ylabel('Unit price', fontsize=6)
axs[0].tick_params(labelsize=6)
axs[0].legend(fontsize=4, loc='upper right')

axs[1].plot(np.arange(0, len(origin_miss[2]), 1), origin_miss[2], c='blue', label='origin')
axs[1].plot(np.arange(0, len(em_miss[2]), 1), em_miss[2], c='magenta', label='EM-algorithm')
axs[1].set_xlabel('Number', fontsize=6)
axs[1].set_ylabel('Quantity', fontsize=6)
axs[1].tick_params(labelsize=6)
axs[1].legend(fontsize=4, loc='upper right')

axs[2].plot(np.arange(0, len(origin_miss[3]), 1), origin_miss[3], c='blue', label='origin')
axs[2].plot(np.arange(0, len(em_miss[3]), 1), em_miss[3], c='magenta', label='EM-algorithm')
axs[2].set_xlabel('Number', fontsize=6)
axs[2].set_ylabel('Tax 5%', fontsize=6)
axs[2].tick_params(labelsize=6)
axs[2].legend(fontsize=4, loc='upper right')

axs[3].plot(np.arange(0, len(origin_miss[4]), 1), origin_miss[4], c='blue', label='origin')
axs[3].plot(np.arange(0, len(em_miss[4]), 1), em_miss[4], c='magenta', label='EM-algorithm')
axs[3].set_xlabel('Number', fontsize=6)
axs[3].set_ylabel('Total', fontsize=6)
axs[3].tick_params(labelsize=6)
axs[3].legend(fontsize=4, loc='upper right')

fig.tight_layout(h_pad=0.2)
plt.show()
