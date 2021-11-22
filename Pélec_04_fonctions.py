import pandas as pd 
import numpy as np
import plotly.express as px

# graphique visualisation vecteurs et données
def visuPCA(df, pca, components, loadings, axis, color=None):
    for f1, f2 in axis:
        if color is None:
            fig = px.scatter(components, x=f1, y=f2)
        else:
            fig = px.scatter(components,
                             x=f1,
                             y=f2,
                             color=color)#,
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