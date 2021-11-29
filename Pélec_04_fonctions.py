import pandas as pd
import numpy as np
import plotly.express as px


# graphique visualisation vecteurs et données
def visuPCA(df, pca, components, loadings, axis, color=None):
    for f1, f2 in axis:
        if color is None:
            fig = px.scatter(components, x=f1, y=f2)
        else:
            fig = px.scatter(components, x=f1, y=f2, color=color)  #,
            #labels={'color': 'Score<br>Nutriscore'})
    for i, feature in enumerate(df.columns):
        fig.add_shape(type='line',
                      x0=0,
                      y0=0,
                      x1=loadings[i, f1] * 10,
                      y1=loadings[i, f2] * 10,
                      line=dict(color='yellow'))
        fig.add_annotation(x=loadings[i, f1] * 10,
                           y=loadings[i, f2] * 10,
                           ax=0,
                           ay=0,
                           xanchor="center",
                           yanchor="bottom",
                           text=feature,
                           bgcolor='white')
    fig.update_layout(
        title='PCA F{} et F{}'.format(f1 + 1, f2 + 1),
        xaxis_title='F{} '.format(f1 + 1) + '(' + str(
            (pca.explained_variance_ratio_[f1] * 100).round(2)) + '%' + ')',
        yaxis_title='F{} '.format(f2 + 1) + '(' + str(
            (pca.explained_variance_ratio_[f2] * 100).round(2)) + '%' + ')')
    return (fig)


from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


# gridsearch pour modèles regression, retourne dataframe avec R², RMSE, RMSLE
def reg_modelGrid(model,
                  scaler,
                  X_train,
                  X_test,
                  y_train,
                  y_test,
                  param_grid,
                  score,
                  y_test_name=None,
                  y_pred_name=None):

    pipemod = make_pipeline(scaler, model)
    gridpipemod = GridSearchCV(
        pipemod,
        param_grid=param_grid,  # hyperparamètres à tester
        cv=5,  # nombre de folds de validation croisée
        scoring=score,  # score à optimiser
        n_jobs=-1)
    gridpipemod.fit(X_train, y_train)
    y_pred = gridpipemod.predict(X_test)

    # score gridsearch
    grid_params = gridpipemod.cv_results_[('params')]
    GridParams = pd.DataFrame(grid_params)
    grid_scores = {}
    for i in range(0, 5):
        grid_scores['ScoresSplit{}'.format(i)] = -gridpipemod.cv_results_[
            ('split{}_test_score'.format(i))]
    grid_scores_mean = gridpipemod.cv_results_[('mean_test_score')]
    grid_scores_sd = gridpipemod.cv_results_[('std_test_score')]
    GridScores = pd.DataFrame(grid_scores).join(
        pd.Series(-grid_scores_mean, name='ScoresMean')).join(
            pd.Series(grid_scores_sd, name='ScoresSD'))
    GridModele = GridParams.join(GridScores)

    # meilleurs paramètres
    best_parameters = gridpipemod.best_estimator_.get_params()
    BestParam = {}
    for param_name in param_grid.keys():
        BestParam[param_name] = best_parameters[param_name]
    BestParametres = pd.DataFrame.from_dict(
        BestParam, orient='index',
        columns=[str(model)
                 ]).reset_index().rename(columns={'index': 'paramètre'})

    # score modèle
    scoreR2 = metrics.r2_score(y_test, y_pred)
    scoreRMSE = metrics.mean_squared_error(y_test, y_pred, squared=False)
    scoreMAE = metrics.mean_absolute_error(y_test, y_pred)
    #scoreMAE100 = metrics.mean_absolute_percentage_error(y_test, y_pred)
    #scoreRMSLE = metrics.mean_squared_log_error(y_test, y_pred, squared=False)
    #    return (pd.DataFrame({model: [scoreR2, scoreRMSLE, scoreRMSE]},
    #                         index=['R²', 'RMSLE', 'RMSE']))

    # dataframe erreur
    ScoreModele = pd.DataFrame(
        {str(model): [scoreR2, scoreRMSE, scoreMAE]},
        index=['R²', 'RMSE', 'MAE'])

    # graph pred vs test
    fig = px.scatter(
        x=y_pred.squeeze(),
        y=y_test.squeeze(),
        labels={
            'x': y_pred_name,
            'y': y_test_name
        },
        title=
        'Visualisation des données prédites par le modèle {} vs les données test'
        .format(model))

    return (GridModele, BestParametres, ScoreModele, y_pred, fig)
