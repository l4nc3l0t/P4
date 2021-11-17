import pandas as pd
import numpy as np
import plotly.express as px
from sklearn import preprocessing, metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

BEB = pd.read_csv('BEB.csv')

# remplacement des valeurs de catégories par des entiers
enc = preprocessing.OrdinalEncoder()
BEBCat = BEB.select_dtypes('object').values
BEBCat_enc = enc.fit_transform(BEBCat)

# rempalcements des valeurs des colonnes catégorielles
BEBEnc = pd.concat([
    BEB.drop(columns=BEB.select_dtypes('object').columns),
    pd.DataFrame(BEBCat_enc,
                 columns=BEB.select_dtypes('object').columns.to_list())
],
                   axis=1)

# sélection de la matrice de données et vecteur cible
BEBM = BEBEnc.drop(columns=['SiteEnergyUse(kBtu)', 'TotalGHGEmissions'])
SiteEnergyUse = BEBEnc['SiteEnergyUse(kBtu)']
TotalGHGEmissions = BEBEnc.TotalGHGEmissions
# création de jeux de test et d'entrainement
BEBM_train, BEBM_test, SiteEnergyUse_train, SiteEnergyUse_test = train_test_split(
    BEBM,
    SiteEnergyUse,
    test_size=0.25  # 25% des données dans le jeu de test
)

std_scale = preprocessing.StandardScaler().fit(BEBM_train)
BEBM_train_std = std_scale.transform(BEBM_train)
BEBM_test_std = std_scale.transform(BEBM_test)
""" ridge = linear_model.Ridge()
ridge_coefs = []
ridge_errors = []
for a in np.logspace(0, 10, 500):
    ridge.set_params(alpha=a)
    ridge.fit(BEBM_train_std, SiteEnergyUse_train)
    ridge_coefs.append(ridge.coef_)
    ridge_errors.append(
        np.mean((ridge.predict(BEBM_test_std) - SiteEnergyUse_test)**2))

lasso = linear_model.Lasso(fit_intercept=False)
lasso_coefs = []
lasso_errors = []
for a in np.logspace(-1, 10, 100):
    lasso.set_params(alpha=a)
    lasso.fit(BEBM_train_std, SiteEnergyUse_train)
    lasso_coefs.append(ridge.coef_)
    lasso_errors.append(
        np.mean((ridge.predict(BEBM_test_std) - SiteEnergyUse_test)**2)) """

parameters = {
    'n_estimators': [10, 50, 100, 300, 500],
    'min_samples_leaf': [1, 3, 5, 10],
    'max_features': ['auto', 'sqrt']
}
rfr_search = GridSearchCV(RandomForestRegressor(),
                          param_grid=parameters,
                          verbose=2,
                          cv=5,
                          n_jobs=-1)

rfr_search.fit(BEBM_train_std, SiteEnergyUse_train)

rfr_search.best_params_

np.sqrt(metrics.mean_squared_error(rfr_search.predict(BEBM_test_std), SiteEnergyUse_test))
