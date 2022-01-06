import pandas as pd
import numpy as np
# ----------------------------------------------------------------#
# -------- Nettoyage et sauvegarde du DataFrame nettoyé --------- #
# ----------------------------------------------------------------#


# df = pd.read_csv('input/RAW_recipes.csv')
#
# print(df.columns)
#
# tabAsupp=["contributor_id","submitted","tags","nutrition","n_steps"]        # Attributs sans intérêt
#
# def nettoyage(tabCSV, tabMot):                                              # Supprime les attributs sans intérêt
#     for i in tabMot:
#         tabCSV.pop(i)
#
# nettoyage(df,tabAsupp)
#
# print(df.columns)
#
# tabNan = pd.isna(df)           # Créer un dataFrame avec comme champ True au lieu des champs Nan
#
# tabLigneAModif = tabNan.query('description == True').index.values       # Donne les index des lignes contenant une description vide
# for i in tabLigneAModif:
#     df.drop(i,inplace=True)                                             # Supprime la ligne où la description est vide
#
# tabNan = pd.isna(df)           # Créer un dataFrame avec True au lieu de Nan
#
# tabLigneAModif = tabNan.query('name == True').index.values       # Donne les index des lignes contenant un name vide
# for i in tabLigneAModif:
#     df.drop(i,inplace=True)                                             # Supprime la ligne où le name est vide
#
#
# df.to_csv("recipe.csv", index=False)                      # Créer un dataframe recipe.csv qui correspond à df nettoyé


# -----------------------------------------------------------------------------------------------------#
# -------- Création des fichiers train.csv(70%) et test.csv(30%) grâce au fichier recipe.csv --------- #
# -----------------------------------------------------------------------------------------------------#

# df = pd.read_csv('recipe.csv')          # Fichier à découper en deux

# -------- Calcul des nombres de lignes pour les fichiers

# print(df.shape)
# print((df.shape[0])*0.7)       # Nombre de ligne dans train : 158 660
# print((df.shape[0])-158660)    #Nombre de ligne dans test : 67 997

# -------- Création du fichier train.csv(70%) du recipe.csv soit 158 660 lignes

# df_train=df.head(158660)
# df_train.to_csv("train.csv", index=False)

# -------- Création du fichier test.csv(30%) du recipe.csv soit 67 997 lignes

# df_train=df.tail(67997)
# df_train.to_csv("test.csv", index=False)

# -------- Fichier train.csv et test.csv

#metadata = pd.read_csv('train.csv')
# print(train.shape[0])
#
# test = pd.read_csv('test.csv')
# print(test.shape[0])




# -------- Division du fichier train.csv de 158 660 lignes en 50 000 lignes

# metadata_train=metadata.head(50000)
# metadata_train.to_csv("train2.csv", index=False)

metadata = pd.read_csv('train2.csv')

# -------- Extraction grâce à sklearn

# #Import TfIdfVectorizer from scikit-learn
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
# tfidf = TfidfVectorizer(stop_words='english')
#
#
# #Construct the required TF-IDF matrix by fitting and transforming the data
# tfidf_matrix = tfidf.fit_transform(metadata['ingredients'])
#
# #Output the shape of tfidf_matrix
# print(tfidf_matrix.shape)
#
# ingredient=tfidf.get_feature_names_out()
#
# # --------- Calcule des similarité
#
# # Import linear_kernel
# from sklearn.metrics.pairwise import linear_kernel
#
# # Compute the cosine similarity matrix
# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#
# print(cosine_sim.shape)
# print(cosine_sim[1])
#
#
#
# indices = pd.Series(metadata.index, index=metadata['name']).drop_duplicates()
# print(indices[:10])
#
# # Function that takes the title of a recipe as input and produces the most similar recipes.
# def get_recommendations(name, cosine_sim=cosine_sim):
#     # Get the index of the recipe that matches the title
#     idx = indices[name]
#
#     # Get the pairwsie similarity scores of all recipes with that recipe
#     sim_scores = list(enumerate(cosine_sim[idx]))
#
#     # Sort the recipes based on the similarity scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#
#     # Get the scores of the 10 most similar recettes
#     sim_scores = sim_scores[1:11]
#
#     # Get the movie indices
#     recipe_indices = [i[0] for i in sim_scores]
#
#     # Return the top 10 most similar movies
#     return metadata['name'].iloc[recipe_indices]
#
# print(get_recommendations('cream  of spinach soup'))

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('input/recipe.csv')

#on a tous les mots d'un phrase séparé
#dTest['ingredients'] = dTest['ingredients'].apply(word_extract)
def word_extract(string):
    lwords = []
    stri = "['" 		                                                #definit le carractère qui est au début et qu'on doit supprimer 
    stri2 = "']"		                                                #definit le carractère qui est a la fin et qu'on doit supprimer
    
    lwords = string.split('\', \'') 		#sépare tous les mots délimité par: ', '
    lwords[0] = lwords[0].replace(stri,"") 							# supprime caractère genant du début
    lwords[len(lwords)-1] = lwords[len(lwords)-1].replace(stri2,"") # supprime caractère genant de la fin
    
    return lwords

df['ingredients'] = df['ingredients'].apply(word_extract)

df2 = df.head(50000)
def recherche(stringContenant):
    nbrContenu = 0
    nbrEnPLusDansLaRecette = 0
    
    for i in range(0,len(tab2),1):
        if(tab2[i] in stringContenant):
            nbrContenu = nbrContenu + 1
    #nbrEnPLusDansLaRecette = len(stringContenant)+1
    return nbrContenu - nbrEnPLusDansLaRecette

tab2 = ['water', 'salt', 'boiling potatoes', 'fresh spinach leaves', 'unsalted butter', 'coarse salt', 'fresh ground black pepper', 'nutmeg']

tableauContenu2 = df2['ingredients'].apply(recherche)         #applique la fonction sur tout le df
column2 = range(0,len(tableauContenu2),1)                     #prépare les index

tableauContenu2 = np.c_[tableauContenu2,column2]               #on associe les résulats avec leur index
tableauContenu2.view('i8,i8').sort(order=['f0'], axis = 0)
print(tableauContenu2)

for i in range(len(tableauContenu2)-10,len(tableauContenu2),1):
    print(tableauContenu2[i])
    
print(tab2)
print(df2['ingredients'].iloc[tableauContenu2[len(tableauContenu2)-1][1]])

for i in range(0,len(tab2),1):
        if(tab2[i] in df2['ingredients'].iloc[tableauContenu2[len(tableauContenu2)-2][1]]):
          print(tab2[i] )



print(metadata["ingredients"].iloc[31460])
print(metadata["ingredients"].iloc[18])
