import numpy as np
import tensorflow as tf
import tflearn
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()  # faire appelle a la class LancasterStemmer
import pickle  # Le module pickle Il est utile pour le transferts de données sur un réseau.
import warnings  # Les warning ne sont pas des erreurs mais des soupçons d’erreurs. Le programme peut continuer mais il est possible qu’il s’arrête un peu plus tard et la cause pourrait être un warning déclenché un peu plus tôt.

warnings.filterwarnings("ignore")
import json  # JavaScript Objet Notation) est un langage léger d'échange de données textuelles.

print("Traitement des intentions...... ")
with open('intentions.json', encoding='utf-8') as json_data:
    intentions = json.load(json_data)
mots = []
classes = []#va contenir les tags
documents = []#va contenir les phrases et les tags
ignore_mots = ['?']
print("en boucle à travers les intentions pour les convertir en mots, classes, documents et ignorer des mot....")
for intention in intentions['intentions']:
    for pattern in intention['patterns']:
        m = nltk.word_tokenize(pattern)  # #permet d'extraire de chaque phrase des mots contenus
        mots.extend(m)

        documents.append((m, intention['tag']))#c'est une liste contenant de tuple qui sont composees a leur tour de liste ,qui contient des patterns, et une string contenant le tag
        #[(['Bonjour'], 'salutation'), (['Salut'], 'salutation'), (['ça', 'va', '?'], 'salutation1'), (['comment', 'allez-vous', '?'], 'salutation1'), (['Au', 'revoir'], 'aurevoir'), (['À', 'bientôt'], 'aurevoir'), (['merci'], 'merci'), (['ça', "m'aide", 'beaucoup'], 'merci'), (['comment', 'contacter', 'un', 'professeur', '?'], 'professeur'), (['où', 'chercher', 'un', 'professeur', '?'], 'professeur'), (['comment', 'contacter', 'un', 'lauréat', '?'], 'lauréat'), (['où', 'chercher', 'un', 'lauréat', '?'], 'lauréat'), (['quels', 'sont', 'les', 'débouchés', 'possibles', '?'], 'débouchés'), (['est-ce', "qu'il", 'y', 'a', 'des', 'offres', 'de', 'stages', '?'], 'stages'), (['quels', 'sont', 'les', 'filières', 'existantes', '?'], 'filières'), (['ingénieur'], 'ingénieur'), (['master'], 'master'), (['doctorat'], 'doctorat'), (['quels', 'sont', 'les', 'horaires', "d'activité", 'de', 'la', 'Bibliothèque', '?'], 'horaire'), (['quels', 'sont', 'les', 'conventions', 'et', 'partenariats', 'de', "l'école", '?'], 'convention'), (['quels', 'sont', 'les', 'clubs', 'de', "l'école", '?'], 'club')]
        if intention['tag'] not in classes:
            classes.append(intention['tag'])#on va remplir la liste class avec les tags du fichier json
# print("stemming et suppression des doublons.....")
mots = [stemmer.stem((m.lower())) for m in mots if m not in ignore_mots]#Premierement on va garder que le racine du mot et on supprime les points et rendre les mots en minuscule
mots = sorted(list(set(mots)))#set supprimer les doublons  et puis on transformer le contenu en liste et en fin on va trier la liste
classes = sorted(list(set(classes)))# de meme pour la classe
# print(len(documents), "documents")
# print(len(classes), "classes", classes)
# print(len(mots), "mots après stemming", mots)
# print("créer les données pour notre modèle....")
training = []
output = []
# print("création d’une liste vide pour la sortie.....")
output_vide = [0] * len(classes)
# print("créer du training set, ensemble de mots pour notre modèle.....")

for doc in documents:
    ensemble = []
    pattern_mots = doc[0]#prend le premier element du tuple qui est un pattern exemple ['bonjour']/['salut']/['ça', 'va', '?']
    pattern_mots = [stemmer.stem(mot.lower()) for mot in pattern_mots]

    for m in mots:
        ensemble.append(1) if m in pattern_mots else ensemble.append(0)#determiner l'emplacement pattern_mots dans la liste mot
    #['bonjo']=>pattern_mots
    #mots=>['a', 'allez-vous', 'au', 'beaucoup', 'bibliothèqu', 'bientôt', 'bonjo', 'cherch', 'club', 'com', 'contact', 'conv', "d'activité", 'de', 'des', 'doct', 'débouchés', 'est-ce', 'et', 'ex', 'filièr', 'horair', 'ingény', "l'école", 'la', 'lauré', 'les', "m'aide", 'mast', 'merc', 'offr', 'où', 'part', 'poss', 'profess', "qu'il", 'quel', 'revoir', 'salut', 'sont', 'stag', 'un', 'va', 'y', 'à', 'ça']
    #ensemble=>[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    output_row = list(output_vide)
    output_row[classes.index(doc[1])] = 1#derminer l'emplacement du tag dans la liste classes
    training.append([ensemble, output_row])#exprimer le document sous forme binaire pour les reseaux de neuronnes
#     [[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]]

import random

training = np.array(training)
print("création de lists de test et training .....")
train_x = list(training[:, 0])#elle recoit les mots
train_y = list(training[:, 1])#elle recoit les tags

print("créer un réseau neuronal pour que notre chatbot soit contextuel......")
print("réinitialisation des données du graphique......")
tf.compat.v1.reset_default_graph()#renitialisation du graph tensoflow
net = tflearn.input_data(shape=[None, len(train_x[0])])#couche d'entrees et le nombre de ces unitees est egale au nombre de mots
net = tflearn.fully_connected(net, 8)#couche cachee entierement connectee a la couche d'entree possedant 8 unites
net = tflearn.fully_connected(net, 8)#couche cachee entierement connectee a la 1ere couche cachee et possedant 8 unites
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')##couche de sortie et le nombre de ces unitees est egale au nombre de tags//la fonction d'activation softmax est une fonction de transfert d'entree en sortie qui est utilisee dans les modeles de reseaux de neuronnes
net = tflearn.regression(net)# La couche de régression est utilisée dans TFLearn pour appliquer une régression (linéaire ou logistique) à l'entrée fournie.
# print("training.....")
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')#Créez le modèle à partir du réseau créé//Cette classe permet de créer un réseau de neurones profond pouvant générer des séquences
# print("traitement the model......")
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)#entraînez le modèle
# print("sauvegarde du modèle.....")
model.save('model.teflearn')#//sauvegarder le modele
# print("pickle est sauvgardé.....")
pickle.dump({'mots': mots, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open("training_data", 'wb'))#wb indique que le fichier est ouvert en écriture en mode binaire.
#pickle.dump est une fonction pour stocker les données d'objet dans le fichier et dans le fichier training_data.

# print("chargement de pickle..........")
data = pickle.load(open("training_data", "rb"))
mots = data['mots']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']
with open('intentions.json', encoding='utf-8') as json_data:
    intentions = json.load(json_data)
# print("chargement de model......")
#
#
def traitement_expression(expression):
    expression_mots = nltk.word_tokenize(expression)
    expression_mots = [stemmer.stem(mot.lower()) for mot in expression_mots]
    return expression_mots# va retourner une liste de racines mots

#
def ensemble_tab(expression, mots, show_details=False):
    expression_mots = traitement_expression(expression)
    ensemble = [0] * len(mots)
    for e in expression_mots:
        for i, m in enumerate(mots):
            if m == e:
                ensemble[i] = 1
                if show_details:
                    print("c'est trouvé dans l'ensemble: %e" % m)
    return (np.array(ensemble))#Creer une liste table qui determiner si les mots d'une expression sont presents dans la liste des mots
#
#
def classification(expression):
    results = model.predict([ensemble_tab(expression, mots)])[0] #retourne un tableau de probilites
    results = [[i, r] for i, r in enumerate(results)]
    # affecte une liste contenant l'indice du tags de la liste classe et sa probabilitee
    results.sort(key=lambda x: x[1], reverse=True)
    #trie la liste par ordre decroissant des probabilitees
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list #retourner une liste de tags et leurs probabilitees
# [('salutation', 0.9442372), ('professeur', 0.015312727), ('doctorat', 0.014229288), ('ingénieur', 0.010746711), ('lauréat', 0.010314825), ('salutation1', 0.002832498), ('aurevoir', 0.0014958007), ('master', 0.00064996496), ('merci', 0.00014771246), ('filières', 2.169377e-05), ('club', 6.2391305e-06), ('convention', 3.2557648e-06), ('horaire', 8.991234e-07), ('stages', 7.1276787e-07), ('débouchés', 5.484304e-07)]

def response(expression, ID_Utilisateur='123', show_details=False):
    results = classification(expression)
    if results:
        while results:
            for i in intentions['intentions']:
                if i['tag'] == results[0][0]:
                    return print(random.choice(i['responses']))
            results.pop(0)
            print(results)


while True:
    input_data = input("you:")
    answer = response(input_data)
    answer