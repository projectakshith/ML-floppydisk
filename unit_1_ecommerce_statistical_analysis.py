# Unit 1: E-commerce statistical analysis
# Extracted from Appendix A of Team_FloppyDisk_Extended_Final_Report.docx.
# For the original Colab notebook where available, see the matching .ipynb file.

import pandas as pd
import numpy as np

df_kaggle = pd.read_csv('kaggle_ecomm_shipping.csv')
df_aug = pd.read_csv('augmented_ecomm_delivery.csv')

print('Kaggle Dataset Shape:', df_kaggle.shape)
print('Augmented Dataset Shape:', df_aug.shape)

print(df_aug.info())
print(df_aug.describe(include='all'))

target_col = 'customer_presence_score'
print('Summary Statistics of Customer Presence Score:')
print(df_aug['customer_presence_score'].describe())
print('Address Type Distribution:')
print(df_aug['address_type'].value_counts(normalize=True))

num_cols = df_aug.select_dtypes(include=[np.number]).columns
corr_matrix = df_aug[num_cols].corr()
print('Correlation Matrix:')
print(corr_matrix)

print('Missing Values Summary:')
print(df_aug.isnull().sum())
