# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # P4 : Anticipez les besoins en consommation électrique de bâtiments

# %%
import os
import wget
import pandas as pd
from ast import literal_eval
import plotly.express as px

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
# GHGEMissions (MetricTonsCO2e) et TotalGHGEmissions renseignent les mêmes informations
# ainsi que GHGEmissionsIntesity (kgCO2e/ft2) et GHGEmissionsIntensity nous allons les
# renommer de la même manière (2016)
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
# graphique du nombre de données par indicateurs après filtre NaN
px.bar(x=BEBFullClean.shape[0] -
       BEBFullClean.isna().sum().sort_values(ascending=False).values,
       y=BEBFullClean.isna().sum().sort_values(ascending=False).index,
       labels=dict(x='Nombre de données', y='Indicateurs'),
       height=1000,
       width=1000)
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
