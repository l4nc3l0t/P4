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
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


# gridsearch pour modèles regression, retourne dataframe avec R², RMSE, RMSLE
def reg_modelGrid(model, scaler, X_train, X_test, y_train, y_test, yname,
                  param_grid, score):
    if str(
            model
    ) == 'RandomForest()' or 'AdaBoostRegressor()' or 'GradientBoostingRegressor()':
        y_train = y_train.ravel()
    preprocessing = make_column_transformer(
        (scaler, make_column_selector(dtype_include='number')),
        (OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore'),
         make_column_selector(dtype_include=object)))
    pipemod = make_pipeline(preprocessing, model)
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
    if '_log' in yname:
        scoreR2 = metrics.r2_score(y_test, y_pred)
        scoreRMSE = metrics.mean_squared_error(np.exp(y_test),
                                               np.exp(y_pred),
                                               squared=False)
        scoreMAE = metrics.mean_absolute_error(np.exp(y_test), np.exp(y_pred))
        scoreMAE100 = metrics.mean_absolute_percentage_error(
            np.exp(y_test), np.exp(y_pred))
    else:
        scoreR2 = metrics.r2_score(y_test, y_pred)
        scoreRMSE = metrics.mean_squared_error(y_test, y_pred, squared=False)
        scoreMAE = metrics.mean_absolute_error(y_test, y_pred)
        scoreMAE100 = metrics.mean_absolute_percentage_error(y_test, y_pred)
    #scoreRMSLE = metrics.mean_squared_log_error(y_test, y_pred, squared=False)
    #    return (pd.DataFrame({model: [scoreR2, scoreRMSLE, scoreRMSE]},
    #                         index=['R²', 'RMSLE', 'RMSE']))

    # dataframe erreur
    ScoreModele = pd.DataFrame(
        {
            'R²': scoreR2,
            'RMSE': scoreRMSE,
            'MAE': scoreMAE,
            'MAE%': scoreMAE100
        },
        index=[str(model)])

    # graph pred vs test
    if '_log' in yname:
        y_test_name = yname.replace('_log', '_test')
        y_pred_name = yname.replace('_log', '_pred')
        fig = px.scatter(
            x=np.exp(y_pred.squeeze()),
            y=np.exp(y_test.squeeze()),
            labels={
                'x': y_pred_name,
                'y': y_test_name
            },
            title=
            'Visualisation des données de {}<br>prédites par le modèle {}<br>vs les données test'
            .format(yname, model))
    else:
        y_test_name = yname + '_test'
        y_pred_name = yname + '_pred'
        # graph pred vs test
        fig = px.scatter(
            x=y_pred.squeeze(),
            y=y_test.squeeze(),
            labels={
                'x': y_pred_name,
                'y': y_test_name
            },
            title=
            'Visualisation des données de {}<br>prédites par le modèle {}<br>vs les données test'
            .format(yname, model))
    return (GridModele, BestParametres, ScoreModele, y_pred, fig)


import plotly.graph_objects as go


# graph visu RMSE
def visuRMSEGrid(model,
                 modelname,
                 paramx,
                 paramxname,
                 gridscoresy,
                 yname,
                 bestparametres=None,
                 parametre=None):
    # modèle à 1 seul paramètre
    if parametre == None:
        # graph RMSE moyenne ± sd
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
        # graph RMSE pour chaque split
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
            title="RMSE du modèle {}<br>pour la variable {}<br>en fonction de {}"
            .format(modelname, yname, paramxname))
    # modèle à 2 paramètres (1 fixe et graph évolution du 2nd)
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
                "RMSE du modèle {} pour la variable<br>{} avec le paramètre {}={}<br>en fonction de l'hyperparamètre {}"
                .format(modelname, yname,
                        parametre.split('__')[1], i, paramxname))

    return (fig3)


def compareGridModels(modelslist,
                  scaler,
                  X_train,
                  X_test,
                  y_train,
                  y_test,
                  yname,
                  paramlist,
                  score,
                  write_data=False,
                  prefix='',
                  suffix=''):
    Result = dict()
    # pour chaques modèles
    for m, p in zip(modelslist, paramlist):
        modelname = f'{m=}'.split('m=')[1].replace('()', '')
        # utilisation de la fonction pour faire GridSearch par modèle
        GridModel, BestParametres, ScoreModele, \
        y_pred, figPredTest = reg_modelGrid(
            m, scaler, X_train, X_test, y_train, y_test, yname, p, score)
        # résultats sous forme de dictionnaire
        Result['Grid' + modelname] = GridModel
        Result['BestParam' + modelname] = BestParametres
        Result['Score' + modelname] = ScoreModele
        Result[yname + '_pred' + modelname] = y_pred
        # utilisation de la fonction pour visualiser la RMSE
        # pour 1 paramètre de la GridSearch
        if len(BestParametres) == 1:
            FigRMSEGRid = visuRMSEGrid(m, modelname,
                                       list(p.values())[0],
                                       list(p.keys())[0].split('__')[1],
                                       GridModel, yname, None, None)
        else:
            FigRMSEGRid = visuRMSEGrid(m, modelname,
                                       list(p.values())[0],
                                       list(p.keys())[0].split('__')[1],
                                       GridModel, yname, BestParametres,
                                       BestParametres['paramètre'][1])
        FigRMSEGRid.show()
        if write_data is True:
            FigRMSEGRid.write_image('./Figures/{}GraphRMSE{}{}.pdf'.format(
                prefix, modelname, suffix))
        # affiches les meilleurs paramètres
        print(BestParametres)
        # affiches les scores
        print(ScoreModele)
        # visualisation des données prédites vs données test
        figPredTest.show()
        if write_data is True:
            figPredTest.write_image('./Figures/{}TestvsPred{}{}.pdf'.format(
                prefix, modelname, suffix))
    return (Result)

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR


def featureRFECV(X, y, yname):
    scaler = RobustScaler(quantile_range=(10, 90))
    X_scale = scaler.fit_transform(X)
    svr = SVR(kernel='linear')
    rfecv = RFECV(estimator=svr,
                  scoring='neg_root_mean_squared_error',
                  n_jobs=-1,
                  verbose=1)
    rfecv.fit(X_scale, y)
    print("Nombre de features optimal pour {} : {}".format(
        yname, rfecv.n_features_))
    print("Noms des features optimales pour les émissions : {}".format(
        rfecv.get_feature_names_out(X.columns)))
    ListColumnsRFECV = []
    ListColumnsRFECV = rfecv.get_feature_names_out(X.columns)
    ScoresRFECV = pd.DataFrame.from_dict(rfecv.cv_results_).abs()

    paramx = [*range(1, len(rfecv.cv_results_['mean_test_score']) + 1)]
    fig1 = go.Figure([
        go.Scatter(name='RMSE moyenne',
                   x=paramx,
                   y=ScoresRFECV.mean_test_score.dropna(),
                   mode='lines',
                   marker=dict(color='red', size=2),
                   showlegend=True),
        go.Scatter(name='SDup RMSE',
                   x=paramx,
                   y=ScoresRFECV.mean_test_score.dropna() +
                   ScoresRFECV.std_test_score.dropna(),
                   mode='lines',
                   marker=dict(color="#444"),
                   line=dict(width=1),
                   showlegend=False),
        go.Scatter(name='SDdown RMSE',
                   x=paramx,
                   y=ScoresRFECV.mean_test_score.dropna() -
                   ScoresRFECV.std_test_score.dropna(),
                   mode='lines',
                   marker=dict(color="#444"),
                   line=dict(width=1),
                   fillcolor='rgba(68, 68, 68, .3)',
                   fill='tonexty',
                   showlegend=False)
    ])

    fig2 = px.line(ScoresRFECV.dropna(),
                   x=paramx,
                   y=[
                       'split0_test_score', 'split1_test_score',
                       'split2_test_score', 'split3_test_score',
                       'split4_test_score'
                   ])

    figRFECV = go.Figure(data=fig1.data + fig2.data)
    figRFECV.update_xaxes(title='Nombre de feature')
    figRFECV.update_yaxes(title='RMSE')
    figRFECV.update_layout(
        title=
        "RMSE pour la variable {} en fonction<br>du nombre de feature selectionnées<br>par recursive feature elimination (RFE)"
        .format(yname))
    return (ListColumnsRFECV, ScoresRFECV, figRFECV)