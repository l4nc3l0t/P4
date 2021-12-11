1. il est conseillé de créer un nouvel environnement python pour éviter les conflits sinon effectuer uniquement les étape 3. 5. et 6. :

`python3 -m venv ./PyP4/`

2. activer l'environnement :

`source ./PyP4/bin/activate`

3. installer les dépendances:

`pip install -r requiremements.txt`

4. ajouter le noyau pour jupyter :

`./PyP4/bin/python -m ipykernel install --user --name 'PyP4'`

5. ouvrir le notebook de nettoyage

`jupyter notebook ./Pélec_01_notebook.ipynb`

6. changer le noyau dans jupyter : / Noyau / Changer le noyau / PyP4

7. faire tourner le fichier Pélec_01_notebook.ipynb pour obtenir les fichiers .csv nécessaires à la modélisation

8. ouvrir le notebook de modélisation

`jupyter notebook ./Pélec_02_code.ipynb`

6. changer le noyau dans jupyter : / Noyau / Changer le noyau / PyP4