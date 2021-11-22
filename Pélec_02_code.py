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
scaler = RobustScaler()

# %%
# ACP sur toutes les colonnes
numPCA = BEB.select_dtypes('number').drop(columns='DataYear').dropna().values
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
for a1, a2 in [[0,1],[0,2],[0,3], [0,4],[1,2],[1,3], [1,4], [2,3], [2,4]]:
    fig = visuPCA(BEB.select_dtypes('number').drop(columns='DataYear').dropna(),
              pca,
              components,
              loadings, [(a1, a2)],
              color=BEB.select_dtypes('number').drop(columns='DataYear').dropna()['SiteEnergyUse(kBtu)'])
    fig.show('browser')
    if write_data is True:
        fig.write_image('./Figures/PCAF{}F{}.pdf'.format(a1+1, a2+1), width=1100, height=1100)

# %%
# modèle régression linéaire
pipeLR = make_pipeline(scaler, linear_model.LinearRegression())

pipeLR.fit(BEBM_train, SiteEnergyUse_train)

SiteEnergyUse_pred = pipeLR.predict(BEBM_test)

LRr2 = metrics.r2_score(SiteEnergyUse_test, SiteEnergyUse_pred)
print("r2 :", LRr2)
LRrmse = metrics.mean_squared_error(SiteEnergyUse_test, SiteEnergyUse_pred)
print("rmse :", LRrmse)

# modèle kNN
pipekNN = make_pipeline(scaler, KNeighborsRegressor(n_jobs=-1))

# Fixer les valeurs des hyperparamètres à tester
param_grid = {
    'kneighborsregressor__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
}

# optimisation score
score = ['r2', 'neg_root_mean_squared_error']

# Classifieur kNN avec recherche d'hyperparamètre par validation croisée
gridpipekNN = GridSearchCV(
    pipekNN,  # un classifieur kNN
    param_grid,  # hyperparamètres à tester
    cv=5,  # nombre de folds de validation croisée
    scoring=score,  # score à optimiser
    refit='neg_root_mean_squared_error',
    n_jobs=-1
)

# Optimisation du classifieur sur le jeu d'entraînement
gridpipekNN.fit(BEBM_train, SiteEnergyUse_train)

# %%
# Afficher l'hyperparamètre optimal
print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
best_parameters = gridpipekNN.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# %%
# Afficher les performances correspondantes
print("Résultats R² de la validation croisée :")
for mean, std, params in zip(
        gridpipekNN.cv_results_['mean_test_r2'],  # score moyen
        gridpipekNN.cv_results_['std_test_r2'],  # écart-type du score
        gridpipekNN.cv_results_['params']  # valeur de l'hyperparamètre
):

    print("{} = {:.3f} (+/-{:.03f}) for {}".format(score[0], mean, std, params))

# Afficher les performances correspondantes
print("Résultats RMSE de la validation croisée :")
for meanrmse, stdrmse, params in zip(
        -(gridpipekNN.cv_results_['mean_test_neg_root_mean_squared_error']),  # score moyen
        gridpipekNN.cv_results_['std_test_neg_root_mean_squared_error'],  # écart-type du score
        gridpipekNN.cv_results_['params']  # valeur de l'hyperparamètre
):

    print("{} = {:.3f} (+/-{:.03f}) for {}".format(score[1], mean, std, params))

# %%
#modèle Rige
piperige = make_pipeline(scaler, linear_model.Ridge())

alphas = np.logspace(2, 5, 200)
param_grid = {'ridge__alpha': alphas}

# %%
errors = []
for a in alphas:
    piperige.set_params(ridge__alpha=a)
    piperige.fit(BEBM_train, SiteEnergyUse_train)
    errors.append(
        np.mean((piperige.predict(BEBM_test) - SiteEnergyUse_test)**2))
# graph rmse
ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.axis('tight')
plt.show()

# %%
# Validation croisée
gridpiperige = GridSearchCV(piperige,
                            param_grid,
                            cv=5,
                            scoring=score,
                            refit='neg_root_mean_squared_error',
                            n_jobs=-1)

gridpiperige.fit(BEBM_train, SiteEnergyUse_train)

# %%
# Afficher l'hyperparamètre optimal
print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
best_parameters = gridpiperige.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# %%
# Afficher les performances correspondantes
print("Résultats pour R² de la validation croisée :")
for meanr2, stdr2, params in zip(
        gridpiperige.cv_results_['mean_test_r2'],  # score moyen
        gridpiperige.cv_results_['std_test_r2'],  # écart-type du score
        gridpiperige.cv_results_['params']  # valeur de l'hyperparamètre
):

    print("{} = {:.5f} (+/-{:.05f}) for {}".format(score[0], mean, std, params))

print("Résultats pour RMSE de la validation croisée :")
for meanrmse, stdrmse, params in zip(
        -(gridpiperige.cv_results_['mean_test_neg_root_mean_squared_error']),  # score moyen
        gridpiperige.cv_results_['std_test_neg_root_mean_squared_error'],  # écart-type du score
        gridpiperige.cv_results_['params']  # valeur de l'hyperparamètre
):

    print("{} = {:.5f} (+/-{:.05f}) for {}".format(score[1], mean, std, params))

# %%
# Prédiction y
SiteEnergyUse_pred = gridpiperige.predict(BEBM_test)

# Calcul RMSE
rmse = metrics.mean_squared_error(SiteEnergyUse_test,
                                  SiteEnergyUse_pred,
                                  squared=False)
print('rmse :', rmse)

# %%
# modèle elastic net
pipeEN = make_pipeline(scaler, linear_model.ElasticNet())

alphas = np.logspace(-3, 3, 1000)
param_grid = {'elasticnet__alpha': alphas}

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
plt.ylabel('error')
plt.axis('tight')
plt.show()

# %%
# Validation croisée
gridpipeEN = GridSearchCV(pipeEN,
                          param_grid,
                          cv=5,
                          scoring=score,
                          refit='neg_root_mean_squared_error',
                          n_jobs=-1)

gridpipeEN.fit(BEBM_train, SiteEnergyUse_train)

# %%
# Afficher l'hyperparamètre optimal
print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
best_parameters = gridpipeEN.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# %%
# Afficher les performances correspondantes
print("Résultats pour R² de la validation croisée :")
for meanr2, stdr2, params in zip(
        gridpipeEN.cv_results_['mean_test_r2'],  # score moyen
        gridpipeEN.cv_results_['std_test_r2'],  # écart-type du score
        gridpipeEN.cv_results_['params']  # valeur de l'hyperparamètre
):

    print("{} = {:.5f} (+/-{:.05f}) for {}".format(score[0], meanr2, stdr2,
                                                   params))

print("Résultats pour RMSE de la validation croisée :")
for meanrmse, stdrmse, params in zip(
        -(gridpipeEN.cv_results_['mean_test_neg_root_mean_squared_error']
          ),  # score moyen
        gridpipeEN.cv_results_[
            'std_test_neg_root_mean_squared_error'],  # écart-type du score
        gridpipeEN.cv_results_['params']  # valeur de l'hyperparamètre
):

    print("{} = {:.5f} (+/-{:.05f}) for {}".format(score[1], meanrmse, stdrmse,
                                                   params))

# %%
# Prédiction y
SiteEnergyUse_pred = gridpipeEN.predict(BEBM_test)

# Calcul RMSE
rmse = metrics.mean_squared_error(SiteEnergyUse_test,
                                  SiteEnergyUse_pred,
                                  squared=False)
print('rmse :', rmse)


# %%
