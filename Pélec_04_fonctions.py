import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


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
    ScoreModele = pd.DataFrame({'R²': scoreR2, 'RMSE' : scoreRMSE, 'MAE' : scoreMAE},
                               index=[str(model)])

    # graph pred vs test
    fig = px.scatter(
        x=y_pred.squeeze(),
        y=y_test.squeeze(),
        labels={
            'x': y_pred_name,
            'y': y_test_name
        },
        title=
        'Visualisation des données prédites par le modèle {}<br>vs les données test'
        .format(model))

    return (GridModele, BestParametres, ScoreModele, y_pred, fig)


# graph visu RMSE
def visuRMSEGrid(model,
                 modelname,
                 paramx,
                 paramxname,
                 gridscoresy,
                 bestparametres=None,
                 parametre=None):
    # graph visualisation RMSE RandomForestRegressor
    # pour le meilleur paramètre max features
    if parametre == None:
        fig1 = go.Figure([
            go.Scatter(name='RMSE moyenne',
                       x=paramx,
                       y=gridscoresy.ScoresMean,
                       mode='lines',
                       marker=dict(color='red', size=2),
                       showlegend=True),
            go.Scatter(name='SDup RMSE',
                       x=paramx,
                       y=gridscoresy.ScoresMean + gridscoresy.ScoresSD,
                       mode='lines',
                       marker=dict(color="#444"),
                       line=dict(width=1),
                       showlegend=False),
            go.Scatter(name='SDdown RMSE',
                       x=paramx,
                       y=gridscoresy.ScoresMean - gridscoresy.ScoresSD,
                       mode='lines',
                       marker=dict(color="#444"),
                       line=dict(width=1),
                       fillcolor='rgba(68, 68, 68, .3)',
                       fill='tonexty',
                       showlegend=False)
        ])

        fig2 = px.line(gridscoresy,
                       x=paramx,
                       y=[
                           'ScoresSplit0', 'ScoresSplit1', 'ScoresSplit2',
                           'ScoresSplit3', 'ScoresSplit4'
                       ])

        fig3 = go.Figure(data=fig1.data + fig2.data)
        fig3.update_xaxes(type='log', title=paramxname)
        fig3.update_yaxes(title='RMSE')
        fig3.update_layout(
            title="RMSE du modèle {} en fonction de {}".format(modelname, paramxname))

    else:
        for i in bestparametres[str(model)][bestparametres['paramètre'] ==
                                            parametre]:
            fig1 = go.Figure([
                go.Scatter(name='RMSE moyenne',
                           x=paramx,
                           y=gridscoresy.ScoresMean.where(
                               gridscoresy[parametre] == i).dropna(),
                           mode='lines',
                           marker=dict(color='red', size=2),
                           showlegend=True),
                go.Scatter(name='SDup RMSE',
                           x=paramx,
                           y=gridscoresy.ScoresMean.where(
                               gridscoresy[parametre] == i).dropna() +
                           gridscoresy.ScoresSD.where(
                               gridscoresy[parametre] == i).dropna(),
                           mode='lines',
                           marker=dict(color="#444"),
                           line=dict(width=1),
                           showlegend=False),
                go.Scatter(name='SDdown RMSE',
                           x=paramx,
                           y=gridscoresy.ScoresMean.where(
                               gridscoresy[parametre] == i).dropna() -
                           gridscoresy.ScoresSD.where(
                               gridscoresy[parametre] == i).dropna(),
                           mode='lines',
                           marker=dict(color="#444"),
                           line=dict(width=1),
                           fillcolor='rgba(68, 68, 68, .3)',
                           fill='tonexty',
                           showlegend=False)
            ])

            fig2 = px.line(
                gridscoresy.where(gridscoresy[parametre] == i).dropna(),
                x=paramx,
                y=[
                    'ScoresSplit0', 'ScoresSplit1', 'ScoresSplit2',
                    'ScoresSplit3', 'ScoresSplit4'
                ])

            fig3 = go.Figure(data=fig1.data + fig2.data)
            fig3.update_xaxes(type='log', title=paramxname)
            fig3.update_yaxes(title='RMSE')
            fig3.update_layout(
                title=
                "RMSE du modèle {} pour le paramètre<br>{}={}<br>en fonction de l'hyperparamètre {}"
                .format(modelname, parametre, i, paramxname))

    return (fig3)
