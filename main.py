import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import seaborn as sns


# -------- Nettoyage et sauvegarde du DataFrame nettoyé ---------

df = pd.read_csv('input/RAW_recipes.csv')

print(df.columns)

tabAsupp=["contributor_id","submitted","tags","nutrition","n_steps"]        # Attributs sans intérêt

def nettoyage(tabCSV, tabMot):                                              # Supprime les attributs sans intérêt
    for i in tabMot:
        tabCSV.pop(i)

nettoyage(df,tabAsupp)

print(df.columns)

tabNan = pd.isna(df)           # Créer un dataFrame avec True au lieu de Nan

tabLigneAModif = tabNan.query('description == True').index.values       # Donne les index des lignes contenant une description vide
for i in tabLigneAModif:
    df.drop(i,inplace=True)                                             # Supprime la ligne où la description est vide

tabNan = pd.isna(df)           # Créer un dataFrame avec True au lieu de Nan

tabLigneAModif = tabNan.query('name == True').index.values       # Donne les index des lignes contenant un name vide
for i in tabLigneAModif:
    df.drop(i,inplace=True)                                             # Supprime la ligne où le name est vide


df.to_csv("input/recipe.csv", index=False)

# df = pd.read_csv('input/recipe.csv')
# print(df.columns)