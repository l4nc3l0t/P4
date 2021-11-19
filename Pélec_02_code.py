import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC

BEB = pd.read_csv('BEB.csv')

BEBM = BEB.drop(columns=['SiteEnergyUse(kBtu)', 'TotalGHGEmissions'])
SiteEnergyUse = np.array(BEB['SiteEnergyUse(kBtu)']).reshape(-1, 1)
TotalGHGEmissions = np.array(BEB.TotalGHGEmissions).reshape(-1, 1)

BEBM_train, BEBM_test, SiteEnergyUse_train, SiteEnergyUse_test = train_test_split(
    BEBM, SiteEnergyUse, test_size=.2)

pipeLR = make_pipeline(StandardScaler(), linear_model.LinearRegression())

pipeLR.fit(BEBM_train, SiteEnergyUse_train)

SiteEnergyUse_pred = pipeLR.predict(BEBM_test)

r2 = metrics.r2_score(SiteEnergyUse_test, SiteEnergyUse_pred)
print("r2 :", r2)
rmse = metrics.mean_squared_error(SiteEnergyUse_test, SiteEnergyUse_pred)
print("rmse :", rmse)


"""
piperige = make_pipeline(StandardScaler(), linear_model.Ridge())

alphas = np.linspace(10, 20, 50)
param_grid = {'ridge__alpha': alphas}

gridsearch = GridSearchCV(piperige, param_grid, n_jobs=-1)

gridsearch.fit(BEBM, SiteEnergyUse)

print("Best score: %0.3f" % gridsearch.best_score_)

print("Best parameters set:")
best_parameters = gridsearch.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

#ax.plot(alphas, gridsearch.)"""