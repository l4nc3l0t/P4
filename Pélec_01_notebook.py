# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # P4 : Anticipez les besoins en consommation électrique de bâtiments

# %%
import os
import wget
import pandas as pd
import numpy as np
from ast import literal_eval
import plotly.express as px
import plotly.graph_objects as go
import folium
from sklearn.linear_model import LinearRegression

# %%
write_data = True

# True : création d'un dossier Figures et Tableau
# dans lesquels seront créés les éléments qui serviront à la présentation
# et écriture des figures et tableaux dans ces dossier
#
# False : pas de création de dossier ni de figures ni de tableaux

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
# ouverture du fichire csv
# utilise le fichier dans le répertoire si il existe
# sinon récupération avec l'url
BEB2015 = pd.read_csv("2015-building-energy-benchmarking.csv")

# %%
# ouverture du fichire csv
# utilise le fichier dans le répertoire si il existe
# sinon récupération avec l'url
BEB2016 = pd.read_csv("2016-building-energy-benchmarking.csv")

# %% [markdown]
# Liste des colonnes présentes uniquement dans les données de 2015

# %%
# colonnes uniquement dans les données de 2015
BEB2015.columns.difference(BEB2016.columns)

# %% [markdown]
# Liste des colonnes présentes uniquement dans les données de 2016

# %%
# colonnes uniquement dans les données de 2016
BEB2016.columns.difference(BEB2015.columns)

# %% [markdown]
# Les données de location en 2015 sont dans une seule colonne
# on va faire en sorte d'uniformiser avec les colonnes présentes en 2016

# %%
# extraction et ajout des données des dict (literal_eval)
# mise sous forme de serie (pd.Series)
# et suppression de la colonne d'origine
BEB2015 = pd.concat([
    BEB2015.drop(columns='Location'),
    BEB2015.Location.map(literal_eval).apply(pd.Series)
],
                    axis=1)
BEB2015 = pd.concat([
    BEB2015.drop(columns='human_address'),
    BEB2015.human_address.map(literal_eval).apply(pd.Series)
],
                    axis=1)

# %%
# renomages des colonnes pour correspondre aux données de 2016
BEB2015 = BEB2015.rename(
    columns={
        "latitude": "Latitude",
        "longitude": "Longitude",
        "address": "Address",
        "city": "City",
        "state": "State",
        "zip": "ZipCode"
    })

# %%
BEB2015.columns.difference(BEB2016.columns)
# %%
BEB2016.columns.difference(BEB2015.columns)
# %% [markdown]
# GHGEMissions (MetricTonsCO2e) et TotalGHGEmissions renseignent les mêmes
# informations ainsi que GHGEmissionsIntesity (kgCO2e/ft2) et
# GHGEmissionsIntensity nous allons les renommer de la même manière (2016)
# %%
# renomages des colonnes pour correspondre aux données de 2016
BEB2015.rename(
    columns={
        "GHGEmissions(MetricTonsCO2e)": "TotalGHGEmissions",
        "GHGEmissionsIntensity(kgCO2e/ft2)": "GHGEmissionsIntensity",
    },
    inplace=True,
)

# %%
BEB2015.columns.difference(BEB2016.columns)
# %%
BEB2016.columns.difference(BEB2015.columns)
# %%
BEB2016.Comments.unique()

# %% [markdown]
# Pas de commentaire dans les données de 2016
# %%
# sup. col. comments
BEB2016.drop(columns="Comments", inplace=True)
# %%
BEB2015.Comment.unique()
# %% [markdown]
# présence de commentaires dans les données de 2015

# %% [markdown]
# Nous allons vérifier que les types des colonnes correspondent
# entre les deux jeux de données
# %%
# dataframe permettant de comparer les types des colonnes
# dans les deux jeux de donées
pd.DataFrame([BEB2015.dtypes, BEB2016.dtypes])

# %% [markdown]
# Les colonnes latitude, longitude et zipcode de 2015
# ne sont pas reconnues comme des nombres nous allons y remédier
# %%
# lat, log et zip en décimaux
BEB2015[['Latitude', 'Longitude',
         'ZipCode']] = BEB2015[['Latitude', 'Longitude',
                                'ZipCode']].astype('float64')
# %% [markdown]
# Nous allons joindre nos données pour n'avoir qu'un seul fichier
# sur lequel travailler lors des test de modèles
# %%
# jonction des deux jeux de données
BEBFull = BEB2015.merge(BEB2016, how="outer")
# %% [markdown]
# Nous allons voir quelques statistiques sur chacunes de nos colonnes

# %%
# stats sur données catégorielles
StatsCat = BEBFull.describe(exclude='number')
StatsCat

# %%
# stats sur données numériques
StatsNum = BEBFull.describe()
StatsNum

# %% [markdown]
# Nous avons des valeurs de surface de batiment et de parking négatives
# nous allons supprimer ces batiments
# %%
# sup. val < 0 dans les valeurs de surface
BEBFullClean = BEBFull.drop(BEBFull[(BEBFull['PropertyGFABuilding(s)'] < 0)
                                    | (BEBFull.PropertyGFAParking < 0)].index)

# %% [markdown]
# Le batiment le plus haut de Seattle fait 76 étages d'après
# [Wikipédia](https://en.wikipedia.org/wiki/List_of_tallest_buildings_in_Seattle)
# Nous allons vérifier que nous n'avons pas de batiments plus grand
# %%
# Affiche les batiments de plus de 76 étages
BEBFullClean[BEBFullClean.NumberofFloors > 76][[
    'OSEBuildingID', 'NumberofFloors', 'PropertyName', 'PrimaryPropertyType'
]]

# %% [markdown]
# Nous avons bien un batiment de 76 étages mais nous avons aussi une église.
# Cette église ne fait pas 99 étages :
# <https://en.wikipedia.org/wiki/Chinese_Baptist_Church#/media/File:Seattle_-_Chinese_Southern_Baptist_01.jpg>
#
# Nous allons remplacer cette valeur par 1

# %%
# remplace la valeur 99 par 1 dans la colonne NumberofFloors
BEBFullClean.NumberofFloors.replace(99, 1, inplace=True)

# %%
# Affiche les batiments ayant 0 batiments
BEBFullClean[BEBFullClean.NumberofBuildings == 0][[
    'OSEBuildingID', 'NumberofBuildings', 'PropertyName', 'PrimaryPropertyType'
]]

# %% [markdown]
# 92 entrée comporte un nombre de batiments nul cela est aberrant.
# Nous allons remplacer ces valeures nulles par 1

# %%
BEBFullClean.NumberofBuildings.replace(0, 1, inplace=True)

# %%
# graphique du nombre de données
fig = px.bar(x=BEBFullClean.isna().sum().sort_values().index,
             y=BEBFullClean.shape[0] -
             BEBFullClean.isna().sum().sort_values().values,
             labels=dict(x='Indicateurs', y='Nombre de données'),
             title='Nombre de données par colonnes',
             height=550,
             width=900)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/DataNbFull.pdf')

# %% [markdown]
# Nous allons supprimer les colonnes comportant moins de 50% de données
# et celles qui ne nous intéressent pas
# (1 seule valeurs dans les colonnes state/city par exemple)
# %%
BEBFullClean.dropna(
    axis='columns',
    thresh=(BEBFullClean.shape[0] * .5),
    # nombre de valeurs = shape * .1
    # soit 90% de NaN et 10% de valeurs
    inplace=True)

# %%
# liste des valeurs dans la colonne City
BEBFullClean.City.unique()
# %%
BEBFullClean.drop(columns=['State', 'City'], inplace=True)

# %%
BEBFullClean.Neighborhood.unique()
# %%
BEBFullClean.Neighborhood.replace('DELRIDGE NEIGHBORHOODS',
                                  'DELRIDGE',
                                  inplace=True)
BEBFullClean.Neighborhood = BEBFullClean.Neighborhood.map(lambda x: x.upper())
# %%
BEBFullClean.Neighborhood.unique()
# %%
# selection des colonnes de type numérique
columns_num = BEBFullClean.select_dtypes('number')
corr = columns_num.corr()
corr = corr.where(np.tril(np.ones(corr.shape)).astype('bool'))
# heatmap à partir de cette matrice
fig = px.imshow(corr, height=700, width=700, color_continuous_scale='balance')
fig.update_layout(plot_bgcolor='white')
fig.show()
if write_data is True:
    fig.write_image('./Figures/HeatmapNum.pdf')

# %% [markdown]
# Les colonnes NaturalGas(kBtu)/NaturalGas(therms) et
# Electricity(kBtu)/Electricity(kWh) sont des doublon l'une de l'autre nous
# n'allons garder que les données en kBtu car c'est l'unité utilisée pour les
# autres indicateurs
#
# Les colonnes SiteEUIWN(kBtu/sf) et SourceEUIWN(kBtu/sf) sont les données
# normalisée en fonction des conditions climatiques moyennes sur 30 ans.
# Elles sont très corrélées au données non normalisées (>.99) nous allons
# donc les supprimer. Les données SiteEnergyUseWN(kBtu) sont moins
# corrélée à SiteEnergyUse(kBtu) (.8) mais nous allons les supprimer
# aussi pour n'avoir que les données brutes
# %%
BEBFullClean.drop(columns=[
    'NaturalGas(therms)', 'Electricity(kWh)', 'SiteEUIWN(kBtu/sf)',
    'SourceEUIWN(kBtu/sf)', 'SiteEnergyUseWN(kBtu)'
],
                  inplace=True)

# %%
for i in BEBFullClean.loc[:,
                          BEBFullClean.columns.str.
                          contains('kbtu', case=False)].drop(
                              columns='SiteEnergyUse(kBtu)').columns.to_list():
    LR = BEBFullClean.loc[:, (i, 'SiteEnergyUse(kBtu)')].dropna()
    lr = LinearRegression()
    X = np.array(LR.iloc[:, 0]).reshape(-1, 1)
    y = LR['SiteEnergyUse(kBtu)']

    LinReg = lr.fit(X, y)
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = LinReg.predict(x_range.reshape(-1, 1))

    rsqrt = LinReg.score(X, y)
    print(rsqrt)

    fig = px.scatter(LR, x=i, y='SiteEnergyUse(kBtu)', opacity=.1)
    fig.add_traces(
        go.Scatter(x=x_range,
                   y=y_range,
                   name=(str(LinReg.coef_[0].round(2)) + 'x' + '<br>R² = ' +
                         str(rsqrt.round(3)))))
    fig.update_layout(legend=dict(yanchor="top",
                                  y=0.99,
                                  xanchor="left",
                                  x=0.01),
                      title='Régression')
    fig.show()

# %%
# graphique du nombre de données par indicateurs après filtre NaN
fig = px.bar(x=BEBFullClean.isna().sum().sort_values().index,
             y=BEBFullClean.shape[0] -
             BEBFullClean.isna().sum().sort_values().values,
             labels=dict(x='Indicateurs', y='Nombre de données'),
             title='Nombre de données par colonnes',
             height=400,
             width=700)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/DataNbDrop.pdf')
# %%
map = folium.Map(
    location=[BEBFullClean.Latitude.mean(),
              BEBFullClean.Longitude.mean()])
for i in range(0, len(BEBFullClean)):
    folium.Circle(location=[
        BEBFullClean.iloc[i].Latitude, BEBFullClean.iloc[i].Longitude
    ],
                  radius=10,
                  popup=BEBFullClean.iloc[i].PropertyName,
                  fill=True).add_to(map)
map

# %% [markdown]
# Les catégories qui vont nous intéresser pour l'entrainement des modèles
# sont les catégories permettant de classifier les batiments à savoir
# BuildingType, PrimaryPropertyType, Neighborhood et LargestPropertyUseType.
# Nous retirons les autres catégories de la matrices que nous utiliserons
# pour l'entrainement des modèles
# %%
useful_cat = [
    'BuildingType', 'PrimaryPropertyType', 'LargestPropertyUseType',
    'Neighborhood'
]
#%%
# selection des données numériques n'étant pas des relevés de consommation
usefull_num = BEBFullClean.select_dtypes('number').drop(columns=[
    'OSEBuildingID', 'CouncilDistrictCode', 'YearBuilt',
    'GHGEmissionsIntensity', 'Latitude', 'Longitude', 'ZipCode'
])
usefull_num = usefull_num.loc[:, ~usefull_num.columns.str.
                            contains('kbtu', case=False)].join(
                                BEBFull['SiteEnergyUse(kBtu)'])

# %%
# heatmap à partir des colonnes numériques utiles
usednum_corr = usefull_num.corr()
usednum_corr = usednum_corr.where(
    np.tril(np.ones(usednum_corr.shape)).astype('bool'))
fig = px.imshow(usednum_corr,
                height=500,
                width=500,
                color_continuous_scale='balance')
fig.update_layout(plot_bgcolor='white')
fig.show()
if write_data is True:
    fig.write_image('./Figures/HeatmapUsedNum.pdf')

# %%
# création dataframe pour étudier les émissions de CO2 et la consommation
# totale d’énergie
BEBClean = usefull_num.drop(columns='ENERGYSTARScore')
BEBClean.dropna(inplace=True)
if write_data is True:
    BEBClean.to_csv('BEB.csv', index=False)

# %%
# création dataframe pour étudier EnergyStarScore
BEBESSClean = usefull_num.dropna()
if write_data is True:
    BEBESSClean.to_csv('BEBESS.csv', index=False)

# %%
