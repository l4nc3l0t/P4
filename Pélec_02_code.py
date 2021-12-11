# %%
import os
import pandas as pd
pd.options.plotting.backend = 'plotly'
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, \
                             GradientBoostingRegressor

from Pélec_04_fonctions import reg_modelGrid, visuRMSEGrid, compareGridModels

# %%
write_data = True

if write_data is True:
    try:
        os.mkdir("./Figures/")
    except OSError as error:
        print(error)
    try:
        os.mkdir("./Tableaux/")
    except OSError as error:
        print(error)
else:
    print("""Visualisation uniquement dans le notebook
    pas de création de figures ni de tableaux""")

# %%
BEBNum = pd.read_csv('BEBNum.csv')

BEBNumM = BEBNum.drop(columns=['SiteEnergyUse(kBtu)', 'TotalGHGEmissions'])
SiteEnergyUse = np.array(BEBNum['SiteEnergyUse(kBtu)']).reshape(-1, 1)
TotalGHGEmissions = np.array(BEBNum.TotalGHGEmissions).reshape(-1, 1)

BEBNumM_train, BEBNumM_test, TotalGHGEmissions_train, TotalGHGEmissions_test = train_test_split(
    BEBNumM, TotalGHGEmissions, test_size=.2)

score = 'neg_root_mean_squared_error'

# %%
# Scaler moins sensible aux outlier d'après la doc
scaler = RobustScaler(quantile_range=(10, 90))

# %% [markdown]
## 1. Modèle de prédiction sur les émissions (TotalGHGEmissions)
### 1.1 Avec les données numériques uniquement
#### 1.1.1 Émissions brutes

# %% [markdown]
##### 1.1.1.1 Modèle LinearRegression

# %%
# modèle régression linéaire
pipeLR = make_pipeline(scaler, LinearRegression())

pipeLR.fit(BEBNumM_train, TotalGHGEmissions_train)

TotalGHGEmissions_predLR = pipeLR.predict(BEBNumM_test)

LRr2 = metrics.r2_score(TotalGHGEmissions_test, TotalGHGEmissions_predLR)
print("r2 :", LRr2)
LRrmse = metrics.mean_squared_error(TotalGHGEmissions_test,
                                    TotalGHGEmissions_predLR,
                                    squared=False)
print("rmse :", LRrmse)

fig = px.scatter(
    x=TotalGHGEmissions_predLR.squeeze(),
    y=TotalGHGEmissions_test.squeeze(),
    labels={
        'x': f'{TotalGHGEmissions_predLR=}'.partition('=')[0],
        'y': f'{TotalGHGEmissions_test=}'.partition('=')[0]
    },
    title=
    "Visualisation des données d'émissions prédites par le modèle de régression linéaire<br>vs les données test"
)
fig.show()
if write_data is True:
    fig.write_image('./Figures/EmissionsLR.pdf')

# %% [markdown]
##### 1.1.1.2 Comparaison de différents modèles sur les émissions brutes

# %%
paramlistEmissions = [{
    'ridge__alpha': np.logspace(1, 5, 100)
}, {
    'lasso__alpha': np.logspace(1, 3, 100)
}, {
    'elasticnet__alpha': np.logspace(0, 3, 100),
    'elasticnet__l1_ratio': np.linspace(0.1, 1, 6)
}, {
    'kneighborsregressor__n_neighbors':
    np.linspace(1, 100, dtype=int)
}, {
    'randomforestregressor__n_estimators':
    np.logspace(0, 3, 10, dtype=int),
    'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
}, {
    'adaboostregressor__n_estimators':
    np.linspace(1, 100, dtype=int),
    'adaboostregressor__loss': ['linear', 'square', 'exponential']
}, {
    'gradientboostingregressor__n_estimators':
    np.logspace(2, 4, 5, dtype=int),
    'gradientboostingregressor__loss':
    ['squared_error', 'absolute_error', 'huber', 'quantile']
}]

ResultEmissions = compareGridModels([
    Ridge(),
    Lasso(),
    ElasticNet(),
    KNeighborsRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor()
], scaler, BEBNumM_train, BEBNumM_test, TotalGHGEmissions_train,
                                    TotalGHGEmissions_test,
                                    'TotalGHGEmissions', paramlistEmissions,
                                    score, write_data, 'Emissions')

# %% [markdown]
#### 1.1.2 Émissions au log

# %%
TotalGHGEmissions_train_log = np.log(TotalGHGEmissions_train)
TotalGHGEmissions_test_log = np.log(TotalGHGEmissions_test)

# %% [markdown]
##### 1.1.2.1 Modèle LinearRegression

# %%
# modèle régression linéaire
pipeLR = make_pipeline(scaler, LinearRegression())

pipeLR.fit(BEBNumM_train, TotalGHGEmissions_train_log)

TotalGHGEmissions_pred_logLR = pipeLR.predict(BEBNumM_test)

LRr2_log = metrics.r2_score(TotalGHGEmissions_test_log,
                            TotalGHGEmissions_pred_logLR)
print("r2 :", LRr2)
LRrmse_log = metrics.mean_squared_error(TotalGHGEmissions_test_log,
                                        TotalGHGEmissions_pred_logLR,
                                        squared=False)
print("rmse :", LRrmse)

fig = px.scatter(
    x=TotalGHGEmissions_pred_logLR.squeeze(),
    y=TotalGHGEmissions_test_log.squeeze(),
    labels={
        'x': f'{TotalGHGEmissions_pred_logLR=}'.partition('=')[0],
        'y': f'{TotalGHGEmissions_test_log=}'.partition('=')[0]
    },
    title=
    "Visualisation des données d'émissions prédites par le modèle de régression linéaire<br>vs les données test"
)
fig.show()
if write_data is True:
    fig.write_image('./Figures/EmissionsLR_log.pdf')

# %% [markdown]
##### 1.1.2.2 Comparaison des modèles sur les émissions au log

# %%
paramlistEmissions_log = [{
    'ridge__alpha': np.logspace(3, 5, 100)
}, {
    'lasso__alpha': np.logspace(-2, 0, 100)
}, {
    'elasticnet__alpha': np.logspace(-1, 1, 10),
    'elasticnet__l1_ratio': np.linspace(0.1, 1, 6)
}, {
    'kneighborsregressor__n_neighbors':
    np.linspace(1, 100, dtype=int)
}, {
    'randomforestregressor__n_estimators':
    np.logspace(0, 3, 10, dtype=int),
    'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
}, {
    'adaboostregressor__n_estimators':
    np.linspace(1, 100, dtype=int),
    'adaboostregressor__loss': ['linear', 'square', 'exponential']
}, {
    'gradientboostingregressor__n_estimators':
    np.logspace(3, 4, 5, dtype=int),
    'gradientboostingregressor__loss':
    ['squared_error', 'absolute_error', 'huber', 'quantile']
}]

ResultEmissions_log = compareGridModels([
    Ridge(),
    Lasso(),
    ElasticNet(),
    KNeighborsRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor()
], scaler, BEBNumM_train, BEBNumM_test, TotalGHGEmissions_train_log,
                                        TotalGHGEmissions_test_log,
                                        'TotalGHGEmissions_log',
                                        paramlistEmissions_log, score,
                                        write_data, 'Emissions', '_log')

# %%
EmissionsScores = pd.DataFrame().append(
    [val for key, val in ResultEmissions.items() if key.startswith('Score')])

# %%
EmissionsScoresLog = pd.DataFrame().append([
    val for key, val in ResultEmissions_log.items() if key.startswith('Score')
]).rename('{}_log'.format)

# %%
EmissionsCompareScores = EmissionsScores.append(EmissionsScoresLog)
if write_data is True:
    EmissionsCompareScores.to_latex('./Tableaux/EmissionsScoresModèles.tex')
EmissionsCompareScores

# %%
fig = make_subplots(len(EmissionsScores.columns),
                    2,
                    column_titles=("Émissions brutes", "Émissions log"),
                    row_titles=(EmissionsScores.columns.to_list()),
                    shared_xaxes=True)
for r, c in enumerate(EmissionsScores):
    fig.add_trace(go.Bar(x=EmissionsScores.index, y=EmissionsScores[c]),
                  row=r + 1,
                  col=1)
    fig.add_trace(go.Bar(x=EmissionsScoresLog.index, y=EmissionsScoresLog[c]),
                  row=r + 1,
                  col=2)
fig.update_layout(title_text="Comparaison des scores des modèles d'émissions",
                  showlegend=False,
                  height=700)
fig.show()
if write_data is True:
    fig.write_image('./Figures/EmissionsCompareScores.pdf', height=700)

# %% [markdown]
#Afin de voir si l'energy star score permet d'améliorer le modèle nous allons
#voir si le meilleurs modèle est amélioré avec cette variable.
#Je choisi d'utiliser le modèle GradientBoosting avec la variable brute
#car c'est le modèle ayant la RMSE la plus faible

# %%
BEBESSNum = pd.read_csv('BEBESSNum.csv')

BEBESSNumM = BEBESSNum.drop(
    columns={'SiteEnergyUse(kBtu)', 'TotalGHGEmissions'})
SiteEnergyUseESS = np.array(BEBESSNum['SiteEnergyUse(kBtu)']).reshape(-1, 1)
TotalGHGEmissionsESS = np.array(BEBESSNum.TotalGHGEmissions).reshape(-1, 1)

BEBESSNumM_train, BEBESSNumM_test, TotalGHGEmissionsESS_train, TotalGHGEmissionsESS_test = train_test_split(
    BEBESSNumM, TotalGHGEmissionsESS, test_size=.2)

TotalGHGEmissionsESS_train_log = np.log(TotalGHGEmissionsESS_train)
TotalGHGEmissionsESS_test_log = np.log(TotalGHGEmissionsESS_test)

# %%
BestParamEmissionsGB = ResultEmissions[
    'BestParamGradientBoostingRegressor'].set_index('paramètre')
paramlistEmissionsESS = [{
    'gradientboostingregressor__n_estimators': [
        int(BestParamEmissionsGB.
            loc['n_estimators'].values)
    ],
    'gradientboostingregressor__loss': [
        *BestParamEmissionsGB.loc[
            'loss', :].values
    ]
}]
ResultEmissionsESS = compareGridModels([GradientBoostingRegressor()],
                                           scaler,
                                           BEBESSNumM_train,
                                           BEBESSNumM_test,
                                           TotalGHGEmissionsESS_train_log,
                                           TotalGHGEmissionsESS_test_log,
                                           'TotalGHGEmissionsESS_log',
                                           paramlistEmissionsESS,
                                           score,
                                           write_data=write_data,
                                           prefix='EmissionsESS',
                                           suffix='_log',
                                           plotfigRMSE=False)

# %%
EmissionsScoresESS = pd.DataFrame().append([
    val for key, val in ResultEmissionsESS.items()
    if key.startswith('Score')
]).rename('{}_ESS'.format)
CompareScoresESS = EmissionsScores.append(EmissionsScoresESS).drop(
    columns=('FitTime(s)')).loc[[
        'GradientBoostingRegressor()', 'GradientBoostingRegressor()_ESS'
    ]]

# %%
fig = make_subplots(1,
                    len(CompareScoresESS.columns),
                    column_titles=(CompareScoresESS.columns.to_list()))
for c, col in enumerate(CompareScoresESS.columns):
    fig.add_trace(go.Bar(x=CompareScoresESS.index, y=CompareScoresESS[col]),
                  row=1,
                  col=c + 1)
fig.update_layout(
    title_text="Comparaison avec et sans ajout de l'energy score stars",
    showlegend=False)
fig.show()
if write_data is True:
    fig.write_image('./Figures/EmissionsCompareScoresESS.pdf')

# %% [markdown]
## 2. Modèle de prédiction sur la consommation énergétique (SiteEnergyUse)
### 2.1 Avec les données numériques uniquement
#### 2.1.1 Consommation énergétique brute
# %%
BEBNumM_train, BEBNumM_test, SiteEnergyUse_train, SiteEnergyUse_test = train_test_split(
    BEBNumM, SiteEnergyUse, test_size=.2)

# %% [markdown]
##### 2.1.1.1 Modèle LinearRegression

# %%
#modèle régression linéaire
pipeLR = make_pipeline(scaler, LinearRegression())

pipeLR.fit(BEBNumM_train, SiteEnergyUse_train)

SiteEnergyUse_predLR = pipeLR.predict(BEBNumM_test)

LRr2 = metrics.r2_score(SiteEnergyUse_test, SiteEnergyUse_predLR)
print("r2 :", LRr2)
LRrmse = metrics.mean_squared_error(SiteEnergyUse_test,
                                    SiteEnergyUse_predLR,
                                    squared=False)
print("rmse :", LRrmse)

fig = px.scatter(
    x=SiteEnergyUse_predLR.squeeze(),
    y=SiteEnergyUse_test.squeeze(),
    labels={
        'x': f'{SiteEnergyUse_predLR=}'.partition('=')[0],
        'y': f'{SiteEnergyUse_test=}'.partition('=')[0]
    },
    title=
    'Visualisation des données de consommation prédites par le modèle de régression linéaire<br>vs les données test'
)
fig.show()
if write_data is True:
    fig.write_image('./Figures/ConsoLR.pdf')

# %% [markdown]
##### 2.1.1.2 Comparaison des modèles sur la consommation
# %%
paramlistConso = [{
    'ridge__alpha': np.logspace(-3, 5, 100)
}, {
    'lasso__alpha': np.logspace(0.1, 3, 100)
}, {
    'elasticnet__alpha': np.logspace(-3, 3, 200),
    'elasticnet__l1_ratio': np.linspace(0.1, 1, 6)
}, {
    'kneighborsregressor__n_neighbors':
    np.linspace(1, 100, dtype=int)
}, {
    'randomforestregressor__n_estimators':
    np.logspace(0, 3, 10, dtype=int),
    'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
}, {
    'adaboostregressor__n_estimators':
    np.linspace(1, 100, dtype=int),
    'adaboostregressor__loss': ['linear', 'square', 'exponential']
}, {
    'gradientboostingregressor__n_estimators':
    np.logspace(2, 3, 5, dtype=int),
    'gradientboostingregressor__loss':
    ['squared_error', 'absolute_error', 'huber', 'quantile']
}]

ResultConso = compareGridModels([
    Ridge(),
    Lasso(),
    ElasticNet(),
    KNeighborsRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor()
], scaler, BEBNumM_train, BEBNumM_test, SiteEnergyUse_train,
                                SiteEnergyUse_test, 'SiteEnergyUse',
                                paramlistConso, score, write_data, 'Conso')

# %% [markdown]
#### 2.1.2 Consommation énergétique au log

# %%
SiteEnergyUse_train_log = np.log(SiteEnergyUse_train)
SiteEnergyUse_test_log = np.log(SiteEnergyUse_test)

# %% [markdown]
##### 2.1.2.1 Modèle LinearRegression

# %%
# modèle régression linéaire
pipeLR = make_pipeline(scaler, LinearRegression())

pipeLR.fit(BEBNumM_train, SiteEnergyUse_train_log)

SiteEnergyUse_pred_logLR = pipeLR.predict(BEBNumM_test)

LRr2_log = metrics.r2_score(SiteEnergyUse_test_log, SiteEnergyUse_pred_logLR)
print("r2 :", LRr2)
LRrmse_log = metrics.mean_squared_error(SiteEnergyUse_test_log,
                                        SiteEnergyUse_pred_logLR,
                                        squared=False)
print("rmse :", LRrmse)

fig = px.scatter(
    x=SiteEnergyUse_pred_logLR.squeeze(),
    y=SiteEnergyUse_test_log.squeeze(),
    labels={
        'x': f'{SiteEnergyUse_pred_logLR=}'.partition('=')[0],
        'y': f'{SiteEnergyUse_test_log=}'.partition('=')[0]
    },
    title=
    'Visualisation des données de consommation prédites par le modèle de régression linéaire<br>vs les données test'
)
fig.show()
if write_data is True:
    fig.write_image('./Figures/ConsoLR_log.pdf')
# %% [markdown]
##### 2.1.2.2 Comparaison des modèles sur la consommation au log

# %%
paramlistConso_log = [{
    'ridge__alpha': np.logspace(1, 4, 100)
}, {
    'lasso__alpha': np.logspace(-3, 0, 100)
}, {
    'elasticnet__alpha': np.logspace(-3, 1, 100),
    'elasticnet__l1_ratio': np.linspace(0.1, 1, 6)
}, {
    'kneighborsregressor__n_neighbors':
    np.linspace(1, 100, dtype=int)
}, {
    'randomforestregressor__n_estimators':
    np.logspace(0, 3, 10, dtype=int),
    'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
}, {
    'adaboostregressor__n_estimators':
    np.linspace(1, 100, dtype=int),
    'adaboostregressor__loss': ['linear', 'square', 'exponential']
}, {
    'gradientboostingregressor__n_estimators':
    np.logspace(1, 4, 5, dtype=int),
    'gradientboostingregressor__loss':
    ['squared_error', 'absolute_error', 'huber', 'quantile']
}]

ResultConso_log = compareGridModels([
    Ridge(),
    Lasso(),
    ElasticNet(),
    KNeighborsRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor()
], scaler, BEBNumM_train, BEBNumM_test, SiteEnergyUse_train_log,
                                    SiteEnergyUse_test_log,
                                    'SiteEnergyUse_log', paramlistConso_log,
                                    score, write_data, 'Conso', '_log')

# %%
ConsoScores = pd.DataFrame().append(
    [val for key, val in ResultConso.items() if key.startswith('Score')])

# %%
ConsoScoresLog = pd.DataFrame().append([
    val for key, val in ResultConso_log.items() if key.startswith('Score')
]).rename('{}_log'.format)

# %%
ConsoCompareScores = ConsoScores.append(ConsoScoresLog)
if write_data is True:
    ConsoCompareScores.to_latex('./Tableaux/ConsoScoresModèles.tex')
ConsoCompareScores

# %%
fig = make_subplots(len(ConsoScores.columns),
                    2,
                    column_titles=("Consommation brute", "Consommation log"),
                    row_titles=(ConsoScores.columns.to_list()),
                    shared_xaxes=True)
for r, c in enumerate(ConsoScores):
    fig.add_trace(go.Bar(x=ConsoScores.index, y=ConsoScores[c]),
                  row=r + 1,
                  col=1)
    fig.add_trace(go.Bar(x=ConsoScoresLog.index, y=ConsoScoresLog[c]),
                  row=r + 1,
                  col=2)
fig.update_layout(
    title_text="Comparaison des scores des modèles de consommation",
    showlegend=False,
    height=700)
fig.show()
if write_data is True:
    fig.write_image('./Figures/ConsoCompareScores.pdf', height=700)
