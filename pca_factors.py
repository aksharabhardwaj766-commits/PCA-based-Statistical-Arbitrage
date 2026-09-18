import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA

'''Standardize Returns'''

def get_standardize_ret(df):
    scalar = StandardScaler()
    standardized = scalar.fit_transform(df)
    return pd.DataFrame(standardized, columns=df.columns, index=df.index)

def run_pca(standardized_df):
    pca = PCA()
    pca.fit(standardized_df)
    return pca



