# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import metrics
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC

from Pélec_04_fonctions import visuPCA

write_data = True
# %%
BEB = pd.read_csv('BEB.csv')

BEBM = BEB.drop(columns=['SiteEnergyUse(kBtu)', 'TotalGHGEmissions'])
SiteEnergyUse = np.array(BEB['SiteEnergyUse(kBtu)']).reshape(-1, 1)
TotalGHGEmissions = np.array(BEB.TotalGHGEmissions).reshape(-1, 1)

BEBM_train, BEBM_test, SiteEnergyUse_train, SiteEnergyUse_test = train_test_split(
    BEBM, SiteEnergyUse, test_size=.2)

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
pipeLR = make_pipeline(scaler, linear_model.LinearRegression())

pipeLR.fit(BEBM_train, SiteEnergyUse_train)

SiteEnergyUse_pred = pipeLR.predict(BEBM_test)

LRr2 = metrics.r2_score(SiteEnergyUse_test, SiteEnergyUse_pred)
print("r2 :", LRr2)
LRrmse = metrics.mean_squared_error(SiteEnergyUse_test,
                                    SiteEnergyUse_pred,
                                    squared=False)
print("rmse :", LRrmse)

# %%
# modèle kNN
pipekNN = make_pipeline(scaler, KNeighborsRegressor(n_jobs=-1))

# Fixer les valeurs des hyperparamètres à tester
n_neighbors = np.linspace(10, 100, dtype=int)

# %%
errors = []
for n in n_neighbors:
    pipekNN.set_params(kneighborsregressor__n_neighbors=n)
    pipekNN.fit(BEBM_train, SiteEnergyUse_train)
    errors.append(
        metrics.mean_squared_error(SiteEnergyUse_test,
                                   pipekNN.predict(BEBM_test),
                                   squared=False))

# graph rmse
ax = plt.gca()
ax.plot(n_neighbors, errors)
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.axis('tight')
plt.show()

# %%
param_grid = {'kneighborsregressor__n_neighbors': n_neighbors}
# optimisation score
score = ['r2', 'neg_mean_squared_error']

# Classifieur kNN avec recherche d'hyperparamètre par validation croisée
gridpipekNN = GridSearchCV(
    pipekNN,  # un classifieur kNN
    param_grid,  # hyperparamètres à tester
    cv=5,  # nombre de folds de validation croisée
    scoring=score,  # score à optimiser
    refit='neg_mean_squared_error',
    n_jobs=-1)

# Optimisation du classifieur sur le jeu d'entraînement
gridpipekNN.fit(BEBM_train, SiteEnergyUse_train)

# %%
# Afficher l'hyperparamètre optimal
print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
best_parameters = gridpipekNN.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# %%
# Erreur kNN
SiteEnergyUse_predkNN = gridpipekNN.predict(BEBM_test)
r2kNN = metrics.r2_score(SiteEnergyUse_test, SiteEnergyUse_predkNN)
print(r2kNN)
rmsekNN = metrics.mean_squared_error(SiteEnergyUse_test,
                                     SiteEnergyUse_predkNN,
                                     squared=False)
print(rmsekNN)

# %%
#modèle Rige
piperige = make_pipeline(scaler, linear_model.Ridge())

alphas = np.logspace(-3, 5, 1000)

# %%
# Validation croisée
param_grid = {'ridge__alpha': alphas}
gridpiperige = GridSearchCV(piperige,
                            param_grid,
                            cv=5,
                            scoring=score,
                            refit='neg_mean_squared_error',
                            n_jobs=-1)

gridpiperige.fit(BEBM_train, SiteEnergyUse_train)

# %%
# Afficher l'hyperparamètre optimal
print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
best_parameters = gridpiperige.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# %%
# Erreur rigde
SiteEnergyUse_predrige = gridpiperige.predict(BEBM_test)
r2rige = metrics.r2_score(SiteEnergyUse_test, SiteEnergyUse_predrige)
print(r2rige)
rmserige = metrics.mean_squared_error(SiteEnergyUse_test,
                                      SiteEnergyUse_predrige,
                                      squared=False)
print(rmserige)

# %%
# graph R² en fonction de alpha
scoresR2_mean = gridpiperige.cv_results_[('mean_test_r2')]
scoresR2_sd = gridpiperige.cv_results_[('std_test_r2')]

fig = px.line(x=alphas,
              y=scoresR2_mean,
              log_x=True,
              labels={
                  'x': 'alpha',
                  'y': 'R²'
              },
              title='R² en fonction de alpha')  #, error_y=scoresR2_sd)
fig.show()

# %%
# graph RMSE en fonction de alpha
scoresRMSE_mean = gridpiperige.cv_results_[(
    'mean_test_neg_mean_squared_error')]
scoresRMSE_sd = gridpiperige.cv_results_[('std_test_neg_mean_squared_error')]

fig = px.line(x=alphas,
              y=-scoresRMSE_mean,
              log_x=True,
              labels={
                  'x': 'alpha',
                  'y': 'RMSE'
              },
              title='RMSE en fonction de alpha')  #, error_y=scoresRMSE_mean)
fig.show()

# %%
# modèle elastic net
pipeEN = make_pipeline(scaler, linear_model.ElasticNet())

alphas = np.logspace(-5, 1, 1000)
l1ratio = np.linspace(0, 1, 5)

# %%
errors = []
for a in alphas:
    pipeEN.set_params(elasticnet__alpha=a)
    pipeEN.fit(BEBM_train, SiteEnergyUse_train)
    errors.append(
        metrics.mean_squared_error(SiteEnergyUse_test,
                                   pipeEN.predict(BEBM_test),
                                   squared=False))

# graph rmse
ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.axis('tight')
plt.show()

# %%
# Validation croisée
param_grid = {'elasticnet__alpha': alphas, 'elasticnet__l1_ratio': l1ratio}
gridpipeEN = GridSearchCV(pipeEN,
                          param_grid,
                          cv=5,
                          scoring=score,
                          refit='neg_mean_squared_error',
                          n_jobs=-1)

gridpipeEN.fit(BEBM_train, SiteEnergyUse_train)

# %%
# Afficher l'hyperparamètre optimal
print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
best_parameters = gridpipeEN.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# %%
# Erreur ElasticNet
SiteEnergyUse_predEN = gridpipeEN.predict(BEBM_test)
r2EN = metrics.r2_score(SiteEnergyUse_test, SiteEnergyUse_predEN)
print(r2EN)
rmseEN = metrics.mean_squared_error(SiteEnergyUse_test,
                                    SiteEnergyUse_predEN,
                                    squared=False)
print(rmseEN)

