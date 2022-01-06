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

print(metadata["ingredients"].iloc[31460])
print(metadata["ingredients"].iloc[18])