# %%
import os
import pandas as pd

pd.options.plotting.backend = 'plotly'
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import metrics
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, \
                             GradientBoostingRegressor

from Pélec_04_fonctions import reg_modelGrid, visuRMSEGrid, compareModels

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
# Scaler moins sensible aux outlier d'après la doc
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

# %% [markdown]
##### 1.1.1.2 Comparaison de différents modèles sur les émissions brutes

# %%
# paramètre Ridge
alphasridge = np.logspace(1, 5, 100)
# paramètre Lasso
alphaslasso = np.logspace(0, 3, 100)
# paramètre ElasticNet
alphasEN = np.logspace(0, 3, 100)
l1ratioEN = np.linspace(0.1, 1, 6)
# paramètre kNN
n_neighbors = np.linspace(1, 100, dtype=int)
# paramètre RandomForest
n_estimatorsRF = np.logspace(0, 3, 10, dtype=int)
# paramètre AdaBoost
n_estimatorsAB = np.logspace(0, 2, 30, dtype=int)
# paramètre GradientBoost
n_estimatorsGB = np.logspace(0, 3, 5, dtype=int)
paramlist = [{
    'ridge__alpha': alphasridge
}, {
    'lasso__alpha': alphaslasso
}, {
    'elasticnet__alpha': alphasEN,
    'elasticnet__l1_ratio': l1ratioEN
}, {
    'kneighborsregressor__n_neighbors': n_neighbors
}, {
    'randomforestregressor__n_estimators': n_estimatorsRF,
    'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
}, {
    'adaboostregressor__n_estimators': n_estimatorsAB,
    'adaboostregressor__loss': ['linear', 'square', 'exponential']
}, {
    'gradientboostingregressor__n_estimators':
    n_estimatorsGB,
    'gradientboostingregressor__loss':
    ['squared_error', 'absolute_error', 'huber', 'quantile']
}]
ResultEmissions = compareModels([
    Ridge(),
    Lasso(),
    ElasticNet(),
    KNeighborsRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor()
], RobustScaler(), BEBNumM_train, BEBNumM_test, TotalGHGEmissions_train,
                                TotalGHGEmissions_test, 'TotalGHGEmissions',
                                paramlist, score, write_data, 'Emissions')

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
# %% [markdown]
##### 1.1.2.2 Comparaison des modèles sur les émissions au log

# %%
alphasridge_log = np.logspace(3, 6, 100)

alphaslasso_log = np.logspace(-2, 1, 100)

alphasEN_log = np.logspace(0, 2, 10)
l1ratioEN_log = np.linspace(0.1, 1, 6)

n_neighbors_log = np.linspace(1, 100, dtype=int)

n_estimatorsRF_log = np.logspace(0, 3, 10, dtype=int)

n_estimatorsAB_log = np.logspace(0, 2, 30, dtype=int)

n_estimatorsGB_log = np.logspace(0, 4, 5, dtype=int)

paramlist_log = [{
    'ridge__alpha': alphasridge_log
}, {
    'lasso__alpha': alphaslasso_log
}, {
    'elasticnet__alpha': alphasEN_log,
    'elasticnet__l1_ratio': l1ratioEN_log
}, {
    'kneighborsregressor__n_neighbors': n_neighbors_log
}, {
    'randomforestregressor__n_estimators':
    n_estimatorsRF_log,
    'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
}, {
    'adaboostregressor__n_estimators': n_estimatorsAB_log,
    'adaboostregressor__loss': ['linear', 'square', 'exponential']
}, {
    'gradientboostingregressor__n_estimators':
    n_estimatorsGB_log,
    'gradientboostingregressor__loss':
    ['squared_error', 'absolute_error', 'huber', 'quantile']
}]

ResultEmissions_log = compareModels([
    Ridge(),
    Lasso(),
    ElasticNet(),
    KNeighborsRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor()
], RobustScaler(), BEBNumM_train, BEBNumM_test, TotalGHGEmissions_train_log,
                                    TotalGHGEmissions_test_log,
                                    'TotalGHGEmissions_log', paramlist_log,
                                    score, write_data, 'Emissions', '_log')

# %%
Scores = pd.DataFrame().append(
    [val for key, val in ResultEmissions.items() if key.startswith('Score')])

# %%
ScoresLog = pd.DataFrame().append([
    val for key, val in ResultEmissions_log.items() if key.startswith('Score')
]).rename('{}_log'.format)

# %%
CompareScores = Scores.append(ScoresLog)
if write_data is True:
    CompareScores.to_latex('./Tableaux/EmissionsScoresModèles.tex')
CompareScores

# %%
fig = make_subplots(4,
                    2,
                    column_titles=("Émissions brutes", "Émissions log"),
                    row_titles=('R²', 'RMSE', 'MAE', 'MAE%'),
                    shared_xaxes=True)
fig.add_trace(go.Bar(x=Scores.index, y=Scores['R²']), row=1, col=1)
fig.add_trace(go.Bar(x=Scores.index, y=Scores['RMSE']), row=2, col=1)
fig.add_trace(go.Bar(x=Scores.index, y=Scores['MAE']), row=3, col=1)
fig.add_trace(go.Bar(x=Scores.index, y=Scores['MAE%']), row=4, col=1)
fig.add_trace(go.Bar(x=ScoresLog.index, y=ScoresLog['R²']), row=1, col=2)
fig.add_trace(go.Bar(x=ScoresLog.index, y=ScoresLog['RMSE']), row=2, col=2)
fig.add_trace(go.Bar(x=ScoresLog.index, y=ScoresLog['MAE']), row=3, col=2)
fig.add_trace(go.Bar(x=ScoresLog.index, y=ScoresLog['MAE%']), row=4, col=2)
fig.update_layout(title_text="Comparaison des scores des modèles d'émissions",
                  showlegend=False)
fig.show()
if write_data is True:
    fig.write_image('./Figures/EmissionsCompareScores.pdf')
"""
# %% [markdown]
### 1.2 Avec les données catégorielles
# %%
BEBCat = pd.read_csv('BEBCat.csv')

BEBCatM = BEBCat.drop(columns=['SiteEnergyUse(kBtu)', 'TotalGHGEmissions'])
SiteEnergyUse = np.array(BEBCat['SiteEnergyUse(kBtu)']).reshape(-1, 1)
TotalGHGEmissions = np.array(BEBCat.TotalGHGEmissions).reshape(-1, 1)

BEBCatM_train, BEBCatM_test, TotalGHGEmissionsCat_train, TotalGHGEmissionsCat_test = train_test_split(
    BEBCatM, TotalGHGEmissions, test_size=.2)

score = 'neg_root_mean_squared_error'

"""

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
# %% [markdown]
##### 2.1.1.2 Comparaison des modèles sur la consommation
# %%

alphasridge = np.logspace(-3, 5, 1000)

alphaslasso = np.linspace(0.1, 1, 5)

alphasEN = np.logspace(-3, 3, 200)
l1ratioEN = np.linspace(0, 1, 6)

n_neighbors = np.linspace(1, 100, dtype=int)

n_estimatorsRF = np.logspace(0, 3, 10, dtype=int)

n_estimatorsAB = np.logspace(0, 2, 30, dtype=int)

n_estimatorsGB = np.logspace(1, 3, 10, dtype=int)

ResultConso = compareModels([
    Ridge(),
    Lasso(),
    ElasticNet(),
    KNeighborsRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor()
], RobustScaler(), BEBNumM_train, BEBNumM_test, SiteEnergyUse_train,
                            SiteEnergyUse_test, 'SiteEnergyUse', paramlist,
                            score, write_data, 'Conso')

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
# %% [markdown]
##### 2.1.2.2 Comparaison des modèles sur la consommation au log

# %%

alphasridge_log = np.logspace(-3, 5, 1000)

alphaslasso_log = np.linspace(0.1, 1, 5)

alphasEN_log = np.logspace(-1, 3, 200)
l1ratioEN_log = np.linspace(0, 1, 6)

n_neighbors_log = np.linspace(1, 100, dtype=int)

n_estimatorsRF_log = np.logspace(0, 3, 10, dtype=int)

n_estimatorsAB_log = np.logspace(0, 2, 30, dtype=int)

n_estimatorsGB_log = np.logspace(1, 4, 10, dtype=int)

ResultConso_log = compareModels([
    Ridge(),
    Lasso(),
    ElasticNet(),
    KNeighborsRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor()
], RobustScaler(), BEBNumM_train, BEBNumM_test, SiteEnergyUse_train_log,
                                SiteEnergyUse_test_log, 'SiteEnergyUse_log',
                                paramlist_log, score, write_data, 'Conso',
                                '_log')

# %%
Scores = pd.DataFrame().append(
    [val for key, val in ResultConso.items() if key.startswith('Score')])

# %%
ScoresLog = pd.DataFrame().append([
    val for key, val in ResultConso_log.items() if key.startswith('Score')
]).rename('{}_log'.format)

# %%
fig = make_subplots(3,
                    2,
                    column_titles=("Consommation brute", "Consommation log2"),
                    row_titles=('R²', 'RMSE', 'MAE'),
                    shared_xaxes=True)
fig.add_trace(go.Bar(x=Scores.index, y=Scores['R²']), row=1, col=1)
fig.add_trace(go.Bar(x=Scores.index, y=Scores['RMSE']), row=2, col=1)
fig.add_trace(go.Bar(x=Scores.index, y=Scores['MAE']), row=3, col=1)
fig.add_trace(go.Bar(x=ScoresLog.index, y=ScoresLog['R²']), row=1, col=2)
fig.add_trace(go.Bar(x=ScoresLog.index, y=ScoresLog['RMSE']), row=2, col=2)
fig.add_trace(go.Bar(x=ScoresLog.index, y=ScoresLog['MAE']), row=3, col=2)
fig.update_layout(
    title_text="Comparaison des scores des modèles de consommation",
    showlegend=False)
fig.show()
if write_data is True:
    fig.write_image('./Figures/ConsoCompareScores.pdf')