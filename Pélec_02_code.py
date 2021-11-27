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
from sklearn.ensemble import RandomForestRegressor

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
    fig.show('browser')
    if write_data is True:
        fig.write_image('./Figures/PCAF{}F{}.pdf'.format(a1 + 1, a2 + 1),
                        width=1100,
                        height=1100)

# %%
# modèle régression linéaire
pipeLR = make_pipeline(scaler, LinearRegression())

pipeLR.fit(BEBM_train, SiteEnergyUse_train)

SiteEnergyUse_predLR = pipeLR.predict(BEBM_test)

LRr2 = metrics.r2_score(SiteEnergyUse_test, SiteEnergyUse_pred)
print("r2 :", LRr2)
LRrmse = metrics.mean_squared_error(SiteEnergyUse_test,
                                    SiteEnergyUse_pred,
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
        'Visualisation des données prédites par le modèle de régression linéaire vs les données test')
fig.show()
#%%
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
ScoresRidge
figRidge.show()
# %%
# graph visualisation RMSE Ridge pour tout les paramètres de GridSearchCV
fig1 = go.Figure([
    go.Scatter(name='RMSE moyenne',
               x=alphasridge,
               y=GridRidge.ScoresMean,
               mode='lines',
               marker=dict(color='red', size=2),
               showlegend=True),
    go.Scatter(name='SDup RMSE',
               x=alphasridge,
               y=GridRidge.ScoresMean + GridRidge.ScoresSD,
               mode='lines',
               marker=dict(color="#444"),
               line=dict(width=1),
               showlegend=False),
    go.Scatter(name='SDdown RMSE',
               x=alphasridge,
               y=GridRidge.ScoresMean - GridRidge.ScoresSD,
               mode='lines',
               marker=dict(color="#444"),
               line=dict(width=1),
               fillcolor='rgba(68, 68, 68, .3)',
               fill='tonexty',
               showlegend=False)
])

fig2 = px.line(GridRidge,
               x=alphasridge,
               y=[
                   'ScoresSplit0', 'ScoresSplit1', 'ScoresSplit2',
                   'ScoresSplit3', 'ScoresSplit4'
               ])

fig3 = go.Figure(data=fig1.data + fig2.data)
fig3.update_xaxes(type='log', title='alpha')
fig3.update_yaxes(title='RMSE')
fig3.update_layout(
    title="RMSE du modèle Ridge en fonction de l'hyperparamètre alpha")
fig3.show()
if write_data is True:
    fig3.write_image('./Figures/graphRMSERidge.pdf')
# %%
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
ScoresLasso
figLasso.show()

# %%
# graph visualisation RMSE Lasso pour tout les paramètres de GridSearchCV
fig1 = go.Figure([
    go.Scatter(name='RMSE moyenne',
               x=alphaslasso,
               y=GridLasso.ScoresMean,
               mode='lines',
               marker=dict(color='red', size=2),
               showlegend=True),
    go.Scatter(name='SDup RMSE',
               x=alphaslasso,
               y=GridLasso.ScoresMean + GridLasso.ScoresSD,
               mode='lines',
               marker=dict(color="#444"),
               line=dict(width=1),
               showlegend=False),
    go.Scatter(name='SDdown RMSE',
               x=alphaslasso,
               y=GridLasso.ScoresMean - GridLasso.ScoresSD,
               mode='lines',
               marker=dict(color="#444"),
               line=dict(width=1),
               fillcolor='rgba(68, 68, 68, .3)',
               fill='tonexty',
               showlegend=False)
])

fig2 = px.line(GridLasso,
               x=alphaslasso,
               y=[
                   'ScoresSplit0', 'ScoresSplit1', 'ScoresSplit2',
                   'ScoresSplit3', 'ScoresSplit4'
               ])

fig3 = go.Figure(data=fig1.data + fig2.data)
fig3.update_xaxes(type='log', title='alpha')
fig3.update_yaxes(title='RMSE')
fig3.update_layout(
    title="RMSE du modèle Lasso en fonction de l'hyperparamètre alpha")
fig3.show()
if write_data is True:
    fig3.write_image('./Figures/graphRMSELasso.pdf')

# %%
alphasEN = np.logspace(0, 7, 200)
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
ScoresEN
figEN.show()

# %%
# graph visualisation RMSE ElasticNet pour tout le meilleur paramètre l1 ratio
for i in BestParametresEN['ElasticNet()'][BestParametresEN['paramètre'] ==
                                          'elasticnet__l1_ratio']:
    fig1 = go.Figure([
        go.Scatter(name='RMSE moyenne',
                   x=alphasEN,
                   y=GridEN.ScoresMean.where(
                       GridEN.elasticnet__l1_ratio == i).dropna(),
                   mode='lines',
                   marker=dict(color='red', size=2),
                   showlegend=True),
        go.Scatter(
            name='SDup RMSE',
            x=alphasEN,
            y=GridEN.ScoresMean.where(
                GridEN.elasticnet__l1_ratio == i).dropna() +
            GridEN.ScoresSD.where(GridEN.elasticnet__l1_ratio == i).dropna(),
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=1),
            showlegend=False),
        go.Scatter(
            name='SDdown RMSE',
            x=alphasEN,
            y=GridEN.ScoresMean.where(
                GridEN.elasticnet__l1_ratio == i).dropna() -
            GridEN.ScoresSD.where(GridEN.elasticnet__l1_ratio == i).dropna(),
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=1),
            fillcolor='rgba(68, 68, 68, .3)',
            fill='tonexty',
            showlegend=False)
    ])

    fig2 = px.line(GridEN.where(GridEN.elasticnet__l1_ratio == i).dropna(),
                   x=alphasEN,
                   y=[
                       'ScoresSplit0', 'ScoresSplit1', 'ScoresSplit2',
                       'ScoresSplit3', 'ScoresSplit4'
                   ])

    fig3 = go.Figure(data=fig1.data + fig2.data)
    fig3.update_xaxes(type='log', title='alpha')
    fig3.update_yaxes(title='RMSE')
    fig3.update_layout(
        title=
        "RMSE du modèle EN pour le paramètre l1={:.2} en fonction de l'hyperparamètre alpha"
        .format(i))
    fig3.show()
    if write_data is True:
        fig3.write_image('./Figures/graphRMSEEN{:.2}.pdf'.format(i))

# %%
n_neighbors = np.linspace(0, 100, dtype=int)
param_gridkNN = {'kneighborsregressor__n_neighbors': n_neighbors}


GridkNN, \
BestParametreskNN, \
ScoreskNN, \
SiteEnergyUse_predkNN, \
figkNN = reg_modelGrid(model=KNeighborsRegressor(n_jobs=-1),
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
ScoreskNN
figkNN.show()

# %%
# graph visualisation RMSE kNN pour tout les paramètres de GridSearchCV
fig1 = go.Figure([
    go.Scatter(name='RMSE moyenne',
               x=n_neighbors,
               y=GridkNN.ScoresMean,
               mode='lines',
               marker=dict(color='red', size=2),
               showlegend=True),
    go.Scatter(name='SDup RMSE',
               x=n_neighbors,
               y=GridkNN.ScoresMean + GridkNN.ScoresSD,
               mode='lines',
               marker=dict(color="#444"),
               line=dict(width=1),
               showlegend=False),
    go.Scatter(name='SDdown RMSE',
               x=n_neighbors,
               y=GridkNN.ScoresMean - GridkNN.ScoresSD,
               mode='lines',
               marker=dict(color="#444"),
               line=dict(width=1),
               fillcolor='rgba(68, 68, 68, .3)',
               fill='tonexty',
               showlegend=False)
])

fig2 = px.line(GridkNN,
               x=n_neighbors,
               y=[
                   'ScoresSplit0', 'ScoresSplit1', 'ScoresSplit2',
                   'ScoresSplit3', 'ScoresSplit4'
               ])

fig3 = go.Figure(data=fig1.data + fig2.data)
fig3.update_xaxes(type='log', title='n neighbors')
fig3.update_yaxes(title='RMSE')
fig3.update_layout(
    title=
    "RMSE du modèle kNN en fonction de l'hyperparamètre n le nombre de voisins"
)
fig3.show()
if write_data is True:
    fig3.write_image('./Figures/graphRMSEkNN.pdf')

# %%
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
ScoresRF
figRF.show()

# %%
# graph visualisation RMSE ElasticNet pour tout le meilleur paramètre max features
for i in BestParametresRF['RandomForestRegressor()'][
        BestParametresRF['paramètre'] ==
        'randomforestregressor__max_features']:
    fig1 = go.Figure([
        go.Scatter(
            name='RMSE moyenne',
            x=n_estimatorsRF,
            y=GridRF.ScoresMean.where(
                GridRF.randomforestregressor__max_features == i).dropna(),
            mode='lines',
            marker=dict(color='red', size=2),
            showlegend=True),
        go.Scatter(
            name='SDup RMSE',
            x=n_estimatorsRF,
            y=GridRF.ScoresMean.where(
                GridRF.randomforestregressor__max_features == i).dropna() +
            GridRF.ScoresSD.where(
                GridRF.randomforestregressor__max_features == i).dropna(),
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=1),
            showlegend=False),
        go.Scatter(
            name='SDdown RMSE',
            x=n_estimatorsRF,
            y=GridEN.ScoresMean.where(
                GridRF.randomforestregressor__max_features == i).dropna() -
            GridRF.ScoresSD.where(
                GridRF.randomforestregressor__max_features == i).dropna(),
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=1),
            fillcolor='rgba(68, 68, 68, .3)',
            fill='tonexty',
            showlegend=False)
    ])

    fig2 = px.line(
        GridRF.where(GridRF.randomforestregressor__max_features == i).dropna(),
        x=n_estimatorsRF,
        y=[
            'ScoresSplit0', 'ScoresSplit1', 'ScoresSplit2', 'ScoresSplit3',
            'ScoresSplit4'
        ])

    fig3 = go.Figure(data=fig1.data + fig2.data)
    fig3.update_xaxes(type='log', title='n_estimators')
    fig3.update_yaxes(title='RMSE')
    fig3.update_layout(
        title=
        "RMSE du modèle RF pour le paramètre max_features={} en fonction de l'hyperparamètre alpha"
        .format(i))
    fig3.show()
    if write_data is True:
        fig3.write_image('./Figures/graphRMSERF{}.pdf'.format(i))

# %%
