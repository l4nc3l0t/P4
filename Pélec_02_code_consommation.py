# %%
import os
import pandas as pd

pd.options.plotting.backend = 'plotly'
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn import metrics
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, \
                             GradientBoostingRegressor

from Pélec_04_fonctions import *

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
BEB = pd.read_csv('BEB.csv')

BEBM = BEB.drop(columns=['SiteEnergyUse(kBtu)', 'TotalGHGEmissions'])
SiteEnergyUse = np.array(BEB['SiteEnergyUse(kBtu)']).reshape(-1, 1)
TotalGHGEmissions = np.array(BEB.TotalGHGEmissions).reshape(-1, 1)

BEBM_train, BEBM_test, SiteEnergyUse_train, SiteEnergyUse_test = train_test_split(
    BEBM, SiteEnergyUse, test_size=.2)

score = 'neg_root_mean_squared_error'

# %%
# Scaler moins sensible aux outlier d'après la doc
scaler = RobustScaler(quantile_range=(10, 90))

# %%
# ACP sur toutes les colonnes
numPCA = BEBM.select_dtypes('number').drop(columns='DataYear').dropna().values
RobPCA = make_pipeline(StandardScaler(), PCA())
components = RobPCA.fit_transform(numPCA)
pca = RobPCA.named_steps['pca']
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# %%
# visualisation de la variance expliquée de chaque composante (cumulée)
exp_var_cum = np.cumsum(pca.explained_variance_ratio_)
fig = px.area(x=range(1, exp_var_cum.shape[0] + 1),
              y=exp_var_cum,
              labels={
                  'x': 'Composantes',
                  'y': 'Variance expliquée cumulée'
              })
fig.update_layout(title='Scree plot')
fig.show()
if write_data is True:
    fig.write_image('./Figures/ScreePlot.pdf', height=300)
# %%
# création des graphiques
for a1, a2 in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]:
    fig = visuPCA(
        BEBM.select_dtypes('number').drop(columns='DataYear').dropna(),
        pca,
        components,
        loadings, [(a1, a2)],
        color=None)
    fig.show()
    if write_data is True:
        fig.write_image('./Figures/PCAF{}F{}.pdf'.format(a1 + 1, a2 + 1),
                        width=1100,
                        height=1100)

# %% [markdown]
# # 1. Modèle de prédiction sur la consommation énergétique (SiteEnergyUse)
# ## 1.1 Consommation énergétique brute

# %% [markdown]
# ### 1.1.1 Modèle LinearRegression

# %%
# modèle régression linéaire
pipeLR = make_pipeline(scaler, LinearRegression())

pipeLR.fit(BEBM_train, SiteEnergyUse_train)

SiteEnergyUse_predLR = pipeLR.predict(BEBM_test)

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
    'Visualisation des données de consommation prédites par le modèle de régression linéaire vs les données test'
)
fig.show()

# %% [markdown]
# ### 1.1.2 Modèle Ridge

#%%
# régression ridge
# réglage des paramètre pour la gridsearch
alphasridge = np.logspace(-3, 5, 1000)
param_gridRidge = {'ridge__alpha': alphasridge}

GridRidge, \
BestParametresRidge, \
ScoresRidge, \
SiteEnergyUse_predRidge, \
figRidge = reg_modelGrid(model=Ridge(),
                            scaler=scaler,
                            X_train=BEBM_train,
                            X_test=BEBM_test,
                            y_train=SiteEnergyUse_train,
                            y_test=SiteEnergyUse_test,
                            y_test_name='SiteEnergyUse_test',
                            y_pred_name='SiteEnergyUse_predRidge',
                            score=score,
                            param_grid=param_gridRidge)

print(BestParametresRidge)
print(ScoresRidge)
figRidge.show()
# %%
# graph visualisation RMSE Ridge pour tout les paramètres de GridSearchCV
FigRMSEGRidRidge = visuRMSEGrid(Ridge(), 'Ridge', alphasridge, 'alpha',
                                GridRidge)
FigRMSEGRidRidge.show()
if write_data is True:
    FigRMSEGRidRidge.write_image('./Figures/ConsoGraphRMSERidge.pdf')

# %% [markdown]
# ### 1.1.3 Modèle Lasso

# %%
# régression lasso
# réglage des paramètre pour la gridsearch
alphaslasso = np.linspace(0.1, 1, 5)
param_gridLasso = {'lasso__alpha': alphaslasso}

GridLasso, \
BestParametresLasso, \
ScoresLasso, \
SiteEnergyUse_predLasso, \
figLasso = reg_modelGrid(model=Lasso(),
                            scaler=RobustScaler(quantile_range=(10, 90)),
                            X_train=BEBM_train,
                            X_test=BEBM_test,
                            y_train=SiteEnergyUse_train,
                            y_test=SiteEnergyUse_test,
                            y_test_name='SiteEnergyUse_test',
                            y_pred_name='SiteEnergyUse_predLasso',
                            score=score,
                            param_grid=param_gridLasso)

print(BestParametresLasso)
print(ScoresLasso)
figLasso.show()

# %%
# graph visualisation RMSE Lasso pour tout les paramètres de GridSearchCV
FigRMSEGRidLasso = visuRMSEGrid(Lasso(), 'Lasso', alphaslasso, 'alpha',
                                GridLasso, None, None)
FigRMSEGRidLasso.show()
if write_data is True:
    FigRMSEGRidLasso.write_image('./Figures/ConsoGraphRMSELasso.pdf')

# %% [markdown]
# ### 1.1.4 Modèle ElasticNet

# %%
# régression elasticnet
# réglage des paramètre pour la gridsearch
alphasEN = np.logspace(-3, 3, 200)
l1ratioEN = np.linspace(0, 1, 6)
param_gridEN = {
    'elasticnet__alpha': alphasEN,
    'elasticnet__l1_ratio': l1ratioEN
}

GridEN, \
BestParametresEN, \
ScoresEN, \
SiteEnergyUse_predEN, \
figEN = reg_modelGrid(model=ElasticNet(),
                         scaler=scaler,
                         X_train=BEBM_train,
                         X_test=BEBM_test,
                         y_train=SiteEnergyUse_train,
                         y_test=SiteEnergyUse_test,
                         y_test_name='SiteEnergyUse_test',
                         y_pred_name='SiteEnergyUse_predEN',
                         score=score,
                         param_grid=param_gridEN)

print(BestParametresEN)
print(ScoresEN)
figEN.show()

# %%
# graph visualisation RMSE ElasticNet pour tout le meilleur paramètre l1 ratio
FigRMSEGRidEN = visuRMSEGrid(ElasticNet(), 'EN', alphasEN, 'alpha', GridEN,
                             BestParametresEN, 'elasticnet__l1_ratio')
FigRMSEGRidEN.show()
if write_data is True:
    FigRMSEGRidEN.write_image('./Figures/ConsoGraphRMSEEN.pdf')

# %% [markdown]
# ### 1.1.5 Modèle kNeighborsRegressor

# %%
# modèle kNN
# réglage des paramètre pour la gridsearch
n_neighbors = np.linspace(1, 100, dtype=int)
param_gridkNN = {'kneighborsregressor__n_neighbors': n_neighbors}


GridkNN, \
BestParametreskNN, \
ScoreskNN, \
SiteEnergyUse_predkNN, \
figkNN = reg_modelGrid(model=KNeighborsRegressor(),
                         scaler=scaler,
                         X_train=BEBM_train,
                         X_test=BEBM_test,
                         y_train=SiteEnergyUse_train,
                         y_test=SiteEnergyUse_test,
                         y_test_name='SiteEnergyUse_test',
                         y_pred_name='SiteEnergyUse_predkNN',
                         score=score,
                         param_grid=param_gridkNN)

print(BestParametreskNN)
print(ScoreskNN)
figkNN.show()

# %%
# graph visualisation RMSE kNN pour tout les paramètres de GridSearchCV
FigRMSEGRidkNN = visuRMSEGrid(KNeighborsRegressor(), 'kNN', n_neighbors,
                              'n neighbors', GridkNN)
FigRMSEGRidkNN.show()
if write_data is True:
    FigRMSEGRidkNN.write_image('./Figures/ConsoGraphRMSEkNN.pdf')

# %% [markdown]
# ### 1.1.6 Modèle RandomForestRegressor

# %%
# modèle RandomForestRegressor
# réglage des paramètre pour la gridsearch
n_estimatorsRF = np.logspace(0, 3, 10, dtype=int)
param_gridRF = {
    'randomforestregressor__n_estimators': n_estimatorsRF,
    'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
}

GridRF, \
BestParametresRF, \
ScoresRF, \
SiteEnergyUse_predRF, \
figRF = reg_modelGrid(model=RandomForestRegressor(),
                         scaler=scaler,
                         X_train=BEBM_train,
                         X_test=BEBM_test,
                         y_train=SiteEnergyUse_train.ravel(),
                         y_test=SiteEnergyUse_test,
                         y_test_name='SiteEnergyUse_test',
                         y_pred_name='SiteEnergyUse_predRF',
                         score=score,
                         param_grid=param_gridRF)

print(BestParametresRF)
print(ScoresRF)
figRF.show()

# %%
# graph visualisation RMSE RandomForestRegressor
# pour le meilleur paramètre max features
FigRMSEGRidRF = visuRMSEGrid(RandomForestRegressor(), 'RF', n_estimatorsRF,
                             'n estimators', GridRF, BestParametresRF,
                             'randomforestregressor__max_features')
FigRMSEGRidRF.show()
if write_data is True:
    FigRMSEGRidRF.write_image('./Figures/ConsoGraphRMSERF.pdf')

# %% [markdown]
# ### 1.1.7 Modèle AdaboostRegressor

# %%
# modèle AdaBoostRegressor
# réglage des paramètre pour la gridsearch
n_estimatorsAB = np.logspace(0, 2, 30, dtype=int)
param_gridAB = {
    'adaboostregressor__n_estimators': n_estimatorsAB,
    'adaboostregressor__loss': ['linear', 'square', 'exponential']
}

GridAB, \
BestParametresAB, \
ScoresAB, \
SiteEnergyUse_predAB, \
figAB = reg_modelGrid(model=AdaBoostRegressor(),
                         scaler=scaler,
                         X_train=BEBM_train,
                         X_test=BEBM_test,
                         y_train=SiteEnergyUse_train.ravel(),
                         y_test=SiteEnergyUse_test,
                         y_test_name='SiteEnergyUse_test',
                         y_pred_name='SiteEnergyUse_predAB',
                         score=score,
                         param_grid=param_gridAB)

print(BestParametresAB)
print(ScoresAB)
figAB.show()

# %%
# graph visualisation RMSE AdaBoostRegressor
# pour le meilleur paramètre loss
FigRMSEGRidAB = visuRMSEGrid(AdaBoostRegressor(), 'AB', n_estimatorsAB,
                             'n estimators', GridAB, BestParametresAB,
                             'adaboostregressor__loss')
FigRMSEGRidAB.show()
if write_data is True:
    FigRMSEGRidAB.write_image('./Figures/ConsoGraphRMSEAB.pdf')

# %% [markdown]
# ## 1.2 Consommation énergétique au log

# %%
SiteEnergyUse_train_log = np.log2(1 + SiteEnergyUse_train)
SiteEnergyUse_test_log = np.log2(1 + SiteEnergyUse_test)

# %% [markdown]
# ### 1.2.1 Modèle LinearRegression

# %%
# modèle régression linéaire
pipeLR = make_pipeline(scaler, LinearRegression())

pipeLR.fit(BEBM_train, SiteEnergyUse_train_log)

SiteEnergyUse_pred_logLR = pipeLR.predict(BEBM_test)

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
    'Visualisation des données de consommation prédites par le modèle de régression linéaire vs les données test'
)
fig.show()
# %% [markdown]
# ### 1.2.2 Modèle Ridge

#%%
# régression ridge
# réglage des paramètre pour la gridsearch
alphasridge_log = np.logspace(-3, 5, 1000)
param_gridRidge_log = {'ridge__alpha': alphasridge_log}

GridRidge_log, \
BestParametresRidge_log, \
ScoresRidge_log, \
SiteEnergyUse_pred_logRidge_log, \
figRidge_log = reg_modelGrid(model=Ridge(),
                            scaler=scaler,
                            X_train=BEBM_train,
                            X_test=BEBM_test,
                            y_train=SiteEnergyUse_train_log,
                            y_test=SiteEnergyUse_test_log,
                            y_test_name='SiteEnergyUse_test_log',
                            y_pred_name='SiteEnergyUse_pred_logRidge',
                            score=score,
                            param_grid=param_gridRidge_log)

print(BestParametresRidge_log)
print(ScoresRidge_log)
figRidge_log.show()

# %%
# graph visualisation RMSE Ridge pour tout les paramètres de GridSearchCV
FigRMSEGRidRidge_log = visuRMSEGrid(Ridge(), 'Ridge', alphasridge_log, 'alpha',
                                    GridRidge_log)
FigRMSEGRidRidge_log.show()
if write_data is True:
    FigRMSEGRidRidge_log.write_image(
        './Figures/ConsoGraphRMSERidge_log.pdf')

# %% [markdown]
# ### 1.2.3 Modèle Lasso

# %%
# régression lasso
# réglage des paramètre pour la gridsearch
alphaslasso_log = np.linspace(0.1, 1, 5)
param_gridLasso_log = {'lasso__alpha': alphaslasso_log}

GridLasso_log, \
BestParametresLasso_log, \
ScoresLasso_log, \
SiteEnergyUse_pred_logLasso_log, \
figLasso_log = reg_modelGrid(model=Lasso(),
                            scaler=RobustScaler(quantile_range=(10, 90)),
                            X_train=BEBM_train,
                            X_test=BEBM_test,
                            y_train=SiteEnergyUse_train_log,
                            y_test=SiteEnergyUse_test_log,
                            y_test_name='SiteEnergyUse_test_log',
                            y_pred_name='SiteEnergyUse_pred_logLasso',
                            score=score,
                            param_grid=param_gridLasso_log)

print(BestParametresLasso_log)
print(ScoresLasso_log)
figLasso_log.show()

# %%
# graph visualisation RMSE Lasso pour tout les paramètres de GridSearchCV
FigRMSEGRidLasso_log = visuRMSEGrid(Lasso(), 'Lasso', alphaslasso_log, 'alpha',
                                    GridLasso_log, None, None)
FigRMSEGRidLasso_log.show()
if write_data is True:
    FigRMSEGRidLasso_log.write_image(
        './Figures/ConsoGraphRMSELasso_log.pdf')

# %% [markdown]
# ### 1.2.4 Modèle ElasticNet

# %%
# régression elasticnet
# réglage des paramètre pour la gridsearch
alphasEN_log = np.logspace(-1, 3, 200)
l1ratioEN_log = np.linspace(0, 1, 6)
param_gridEN_log = {
    'elasticnet__alpha': alphasEN_log,
    'elasticnet__l1_ratio': l1ratioEN_log
}

GridEN_log, \
BestParametresEN_log, \
ScoresEN_log, \
SiteEnergyUse_pred_logEN, \
figEN_log = reg_modelGrid(model=ElasticNet(),
                         scaler=scaler,
                         X_train=BEBM_train,
                         X_test=BEBM_test,
                         y_train=SiteEnergyUse_train_log,
                         y_test=SiteEnergyUse_test_log,
                         y_test_name='SiteEnergyUse_test_log',
                         y_pred_name='SiteEnergyUse_pred_logEN',
                         score=score,
                         param_grid=param_gridEN_log)

print(BestParametresEN_log)
print(ScoresEN_log)
figEN_log.show()

# %%
# graph visualisation RMSE ElasticNet pour tout le meilleur paramètre l1 ratio
FigRMSEGRidEN_log = visuRMSEGrid(ElasticNet(), 'EN', alphasEN_log, 'alpha',
                                 GridEN_log, BestParametresEN_log,
                                 'elasticnet__l1_ratio')
FigRMSEGRidEN_log.show()
if write_data is True:
    FigRMSEGRidEN_log.write_image('./Figures/ConsoGraphRMSEEN_log.pdf')

# %% [markdown]
# ### 1.2.5 Modèle kNeighborsRegressor
# %%
# modèle kNN
# réglage des paramètre pour la gridsearch
n_neighbors_log = np.linspace(1, 100, dtype=int)
param_gridkNN_log = {'kneighborsregressor__n_neighbors': n_neighbors_log}


GridkNN_log, \
BestParametreskNN_log, \
ScoreskNN_log, \
SiteEnergyUse_pred_logkNN_log, \
figkNN_log = reg_modelGrid(model=KNeighborsRegressor(),
                         scaler=scaler,
                         X_train=BEBM_train,
                         X_test=BEBM_test,
                         y_train=SiteEnergyUse_train_log,
                         y_test=SiteEnergyUse_test_log,
                         y_test_name='SiteEnergyUse_test_log',
                         y_pred_name='SiteEnergyUse_pred_logkNN',
                         score=score,
                         param_grid=param_gridkNN_log)

print(BestParametreskNN_log)
print(ScoreskNN_log)
figkNN_log.show()

# %%
# graph visualisation RMSE kNN pour les paramètres de GridSearchCV
FigRMSEGRidkNN_log = visuRMSEGrid(KNeighborsRegressor(), 'kNN',
                                  n_neighbors_log, 'n neighbors', GridkNN_log)
FigRMSEGRidkNN_log.show()
if write_data is True:
    FigRMSEGRidkNN_log.write_image('./Figures/ConsoGraphRMSEkNN_log.pdf')

# %% [markdown]
# ### 1.2.6 Modèle RandomForestRegressor

# %%
# modèle RandomForestRegressor
# réglage des paramètre pour la gridsearch
n_estimatorsRF_log = np.logspace(0, 3, 10, dtype=int)
param_gridRF_log = {
    'randomforestregressor__n_estimators': n_estimatorsRF_log,
    'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
}

GridRF_log, \
BestParametresRF_log, \
ScoresRF_log, \
SiteEnergyUse_pred_logRF_log, \
figRF_log = reg_modelGrid(model=RandomForestRegressor(),
                         scaler=scaler,
                         X_train=BEBM_train,
                         X_test=BEBM_test,
                         y_train=SiteEnergyUse_train_log.ravel(),
                         y_test=SiteEnergyUse_test_log,
                         y_test_name='SiteEnergyUse_test_log',
                         y_pred_name='SiteEnergyUse_pred_log_logRF',
                         score=score,
                         param_grid=param_gridRF)

print(BestParametresRF_log)
print(ScoresRF_log)
figRF_log.show()

# %%
# graph visualisation RMSE RandomForestRegressor
# pour le meilleur paramètre max features
FigRMSEGRidRF_log = visuRMSEGrid(RandomForestRegressor(), 'RF',
                                 n_estimatorsRF_log, 'n estimators',
                                 GridRF_log, BestParametresRF_log,
                                 'randomforestregressor__max_features')
FigRMSEGRidRF_log.show()
if write_data is True:
    FigRMSEGRidRF_log.write_image('./Figures/ConsoGraphRMSERF_log.pdf')

# %% [markdown]
# ### 1.2.7 Modèle AdaboostRegressor

# %%
# modèle AdaBoostRegressor
# réglage des paramètre pour la gridsearch
n_estimatorsAB_log = np.logspace(0, 2, 30, dtype=int)
param_gridAB_log = {
    'adaboostregressor__n_estimators': n_estimatorsAB_log,
    'adaboostregressor__loss': ['linear', 'square', 'exponential']
}

GridAB_log, \
BestParametresAB_log, \
ScoresAB_log, \
SiteEnergyUse_pred_logAB, \
figAB_log = reg_modelGrid(model=AdaBoostRegressor(),
                         scaler=scaler,
                         X_train=BEBM_train,
                         X_test=BEBM_test,
                         y_train=SiteEnergyUse_train_log.ravel(),
                         y_test=SiteEnergyUse_test_log,
                         y_test_name='SiteEnergyUse_test_log',
                         y_pred_name='SiteEnergyUse_predAB_log',
                         score=score,
                         param_grid=param_gridAB_log)

print(BestParametresAB_log)
print(ScoresAB_log)
figAB_log.show()

# %%
# graph visualisation RMSE AdaBoostRegressor
# pour le meilleur paramètre loss
FigRMSEGRidAB_log = visuRMSEGrid(AdaBoostRegressor(), 'AB', n_estimatorsAB_log,
                             'n estimators', GridAB_log, BestParametresAB_log,
                             'adaboostregressor__loss')
FigRMSEGRidAB_log.show()
if write_data is True:
    FigRMSEGRidAB_log.write_image('./Figures/ConsoGraphRMSEAB_log.pdf')
        
# %%
Scores = ScoresLasso.join(
    [ScoresRidge, ScoresEN, ScoreskNN, ScoresRF, ScoresAB])

# %%
ScoresLog = ScoresLasso_log.join(
    [ScoresRidge_log, ScoresEN_log, ScoreskNN_log, ScoresRF_log, ScoresAB_log])

# %%
CompareScores = Scores.join(ScoresLog, rsuffix='_log')
if write_data is True:
    CompareScores.to_latex('./Tableaux/ConsoScoresModèles.tex')

# %%
