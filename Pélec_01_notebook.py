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

# %%
