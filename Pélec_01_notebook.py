# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # P4 : Anticipez les besoins en consommation électrique de bâtiments

# %%
import os
import wget
import pandas as pd

# %%
write_data = True

# True : création d'un dossier Figures et Tableau
# dans lesquels seront créés les éléments qui serviront à la présentation
# et écriture des figures et tableaux dans ces dossier
#
# False : pas de création de dossier ni de figures ni de tableaux

if write_data is True:
    try:
        os.mkdir('./Figures/')
    except OSError as error:
        print(error)
    try:
        os.mkdir('./Tableaux/')
    except OSError as error:
        print(error)
else:
    print("""Visualisation uniquement dans le notebook
    pas de création de figures ni de tableaux""")

# %% [markdown]
# Importation des données si nécessaire

# %%
if write_data is True:
    if os.path.exists('BEB2015.csv'):
        print("Le jeu de données est déjà présent dans le répertoire")
    else:
        wget.download("https://data.seattle.gov/api/views/h7rm-fz6m/rows.csv",
                      out='BEB2015.csv')
    if os.path.exists('BEB2016.csv'):
        print("Le jeu de données est déjà présent dans le répertoire")
    else:
        wget.download("https://data.seattle.gov/api/views/2bpz-gwpy/rows.csv",
                      out='BEB2016.csv')

# %%
# ouverture du fichire csv
# utilise le fichier dans le répertoire si il existe
# sinon récupération avec l'url
if os.path.exists('BEB2015.csv'):
    BEB2015 = pd.read_csv('BEB2015.csv')
else:
    BEB2015 = pd.read_csv(
        "https://data.seattle.gov/api/views/h7rm-fz6m/rows.csv")

# %%
# ouverture du fichire csv
# utilise le fichier dans le répertoire si il existe
# sinon récupération avec l'url
if os.path.exists('BEB2015.csv'):
    BEB2016 = pd.read_csv('BEB2016.csv')
else:
    BEB2016 = pd.read_csv(
        "https://data.seattle.gov/api/views/2bpz-gwpy/rows.csv")

# %%
BEB2015.sort_values('OSEBuildingID', inplace=True)
BEB2015.reset_index(inplace=True, drop=True)

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
# Les données de location en 2015 sont dans une seule colonne
# on va faire en sorte d'uniformiser avec les colonnes présentes en 2016

# %%
BEB2015['Latitude'] = BEB2015.Location.str.split(
    '(', expand=True)[1].str.split(',', expand=True)[0].astype('float64')
BEB2015['Longitude'] = BEB2015.Location.str.split(
    '(',
    expand=True)[1].str.split(',',
                              expand=True)[1].str.strip(')').astype('float64')
BEB2015['Address'] = BEB2015.Location.str.split('\n', n=1, expand=True)[0]
BEB2015['City'] = BEB2015.Location.str.split(
    '\n', n=1, expand=True)[1].str.split(',', n=1, expand=True)[0]
BEB2015['State'] = BEB2015.Location.str.split(
    '\n', n=1, expand=True)[1].str.split(
        ',', n=1, expand=True)[1].str.lstrip(' ').str.split(' ',
                                                            n=1,
                                                            expand=True)[0]
BEB2015['ZipCode'] = BEB2015.Location.str.split(
    '\n', n=1, expand=True)[1].str.split(
        ',', n=1, expand=True)[1].str.lstrip(' ').str.split(
            ' ', n=1,
            expand=True)[1].str.split('\n', expand=True)[0].astype('float64')

# %%
BEB2015.drop(columns='Location', inplace=True)

# %%
BEB2015.columns.difference(BEB2016.columns)
# %%
BEB2016.columns.difference(BEB2015.columns)
# %% [markdown]
# GHGEMissions (MetricTonsCO2e) et TotalGHGEmissions renseignent les mêmes informations
# ainsi que GHGEmissionsIntesity (kgCO2e/ft2) et GHGEmissionsIntensity
# %%
BEB2015.rename(columns={
    'GHGEmissions(MetricTonsCO2e)': 'TotalGHGEmissions',
    'GHGEmissionsIntensity(kgCO2e/ft2)': 'GHGEmissionsIntensity'
},
               inplace=True)

# %%
BEB2015.columns.difference(BEB2016.columns)
# %%
BEB2016.columns.difference(BEB2015.columns)
# %%
BEB2016.Comments.unique()

# %% [markdown]
# Pas de commentaire dans les données de 2016
# %%
BEB2016.drop(columns='Comments', inplace=True)
# %%
BEB2015.Comment.unique()
# %% [markdown]
# présence de commentaires dans les données de 2015
# %% [markdown]
# Nous allons joindre nos données pour n'avoir qu'un seul fichier
# sur lequel travailler lors des test de modèles
# %%
BEBFull = BEB2015.merge(BEB2016, how='outer')
# %%
