import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

BEB = pd.read_csv('BEB.csv')

# remplacement des valeurs de catégories par des entiers
enc = preprocessing.OrdinalEncoder()
X = BEB.select_dtypes('object').values
X_enc = enc.fit_transform(X)

# rempalcements des valeurs des colonnes catégorielles
BEBEnc = pd.concat([
    BEB.drop(columns=BEB.select_dtypes('object').columns),
    pd.DataFrame(X_enc, columns=BEB.select_dtypes('object').columns.to_list())
],
                   axis=1)
