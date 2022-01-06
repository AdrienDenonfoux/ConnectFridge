# lien de la BDD : https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions/download
#ou : https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions
choixMethode = 1 #1 recherche d'ingrédients corespondants a ceux rentré , 0 recommandation de recette par rapport a une autre avec un problème de temps d'excusion 

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



if (choixMethode == 1):
	import seaborn as sns
	import sys
	import matplotlib.pyplot as plt
	import seaborn as sns

	df = pd.read_csv('input/recipe.csv')# a changer suivant le dataframe utilisé

	# FONCTION WORD_EXTRACT
	# prend en entré un string, qui est l'ensemble de tous les ingredients qui est donsidéré comme une phrase
	# sépare les mots délimité par : ', '
	# retourne un tableau composé d'ingredients
	# fonction d'exemple pour utiliser : dTest['ingredients'] = dTest['ingredients'].apply(word_extract)
	def word_extract(string):
	    lwords = []
	    stri = "['" 		                                                # definit le carractère qui est au début de chaque mots dans la liste d'ingredient et qu'on doit supprimer 
	    stri2 = "']"		                                                # definit le carractère qui est a la fin de chaque mots dans la liste d'ingredient et qu'on doit supprimer
	    
	    lwords = string.split('\', \'')                                     # sépare tous les mots délimité par: ', '
	    lwords[0] = lwords[0].replace(stri,"")                              # supprime caractère genant du début
	    lwords[len(lwords)-1] = lwords[len(lwords)-1].replace(stri2,"")     # supprime caractère genant de la fin
	    
	    return lwords

	df['ingredients'] = df['ingredients'].apply(word_extract)               # remplacement de la phrase d'ingredient par une liste d'ingrediant

	#a enlever si on ouvre un ficher train ou test et pas le fichier entier
	df2 = df.tail(67997)                                                    # choix de la partie train ou test, 

	# FONCTION RECHERCHE
	# prend en entré un string ou un tableau (ca fonctionne avec les deux) d'ingredient, ici on utilise un tableau ce qui nous permet de trouver le mot exact et pas un qui se rapproche 
	# elle recherche si les ingredients rentré dans ingrediantsPossede[] coresponde avec les ingredients de recettes en paramettre  
	# elle retourne le nombre d' ingrediantsPossede[] contenu dans la recette
	def recherche(tabStringContenant):
	    nbrContenu = 0
	    nbrEnPLusDansLaRecette = 0
	    
	    for i in range(0,len(ingrediantsPossede),1):
	        if(ingrediantsPossede[i] in tabStringContenant):
	            nbrContenu = nbrContenu + 1
	    #nbrEnPLusDansLaRecette = len(tabStringContenant)+1
	    return nbrContenu - nbrEnPLusDansLaRecette

	#Liste d'ingredients possédé
	ingrediantsPossede = ['water', 'salt', 'boiling potatoes', 'fresh spinach leaves', 'unsalted butter', 'coarse salt', 'fresh ground black pepper', 'nutmeg']

	tableauContenu2 = df2['ingredients'].apply(recherche)                   # applique la fonction sur tout le dataframe, c'est donc un tableau de note
	column2 = range(0,len(tableauContenu2),1)                               # création de tabbleau qui corresponde au index des recettes 

	tableauContenu2 = np.c_[tableauContenu2,column2]                        # on associe les résulats des recettes avec leur index, 
	tableauContenu2.view('i8,i8').sort(order=['f0'], axis = 0)              # trie le tableau pour que les recettes contenant le plus d'ingrédients possédé ce retrouve de manière croissante  

	for i in range(len(tableauContenu2)-10,len(tableauContenu2),1):         # affiche le top 10 des recettes 
	    #print(tableauContenu2[i], df2['name'].iloc[tableauContenu2[i][1]])
			print('nom :', df2['name'].iloc[tableauContenu2[i][1]],"\n",'description :', df2['description'].iloc[tableauContenu2[i][1]],"\n",'différentes étapes :', df2['steps'].iloc[tableauContenu2[i][1]],"\n",'temps de préparation :', df2['minutes'].iloc[tableauContenu2[i][1]], 'minutes',"\n",'nombre d\'ingrédients utilisés du frigo :', tableauContenu2[i][0],"\n",'nombre totale d\'ingrédients :', df2['n_ingredients'].iloc[tableauContenu2[i][1]],"\n",'nombre d\'ingredients en plus :',df2['n_ingredients'].iloc[tableauContenu2[i][1]]-tableauContenu2[i][0],"\n")

    
	"""
	print(tab2)                                                                 # tester les ressemblences
	print(df2['ingredients'].iloc[tableauContenu2[len(tableauContenu2)-1][1]])  # tester les ressemblences

	for i in range(0,len(tab2),1):                                              # affichage des ingrédients retenu, pour vérifier.
	        if(tab2[i] in df2['ingredients'].iloc[tableauContenu2[len(tableauContenu2)-1][1]]):
	          print(tab2[i] )
	""" 
else:
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
	# # --------- Calculs des similaritées
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
	#     # Get the scores of the 10 most similar recipes
	#     sim_scores = sim_scores[1:11]
	#
	#     # Get the recipe indices
	#     recipe_indices = [i[0] for i in sim_scores]
	#
	#     # Return the top 10 most similar recipes
	#     return metadata['name'].iloc[recipe_indices]
	#
	# print(get_recommendations('cream  of spinach soup'))

	#print(metadata["ingredients"].iloc[31460])
	#print(metadata["ingredients"].iloc[18])

