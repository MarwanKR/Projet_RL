import pandas as pd 
import json


with open("/Users/elie/Desktop/Polytechnique/RL/Projet/instances/Asmall.json", "r") as f:
    data = json.load(f)

# Afficher la structure
#print(type(data))
#print(data.keys())

df_Asmall = pd.DataFrame(data['trains'])

# Normaliser la colonne 0 en colonnes séparées
df_flat = pd.json_normalize(df_Asmall[0])

# Afficher le DataFrame aplati
print("Trains")
print(df_flat.head())

df_Asmall_itin = pd.DataFrame(data['itineraires'])
print(df_Asmall_itin.head())

#df_interditction = pd.json_normalize(data['interdictionsQuais'])
#print(df_interditction.head())

'''with open("/Users/elie/Desktop/Polytechnique/RL/Projet/instances/inst_PE.json", "r") as f:
    data = json.load(f)

print(data.keys())

df_interditction = pd.json_normalize(data['interdictionsQuais'])
print(df_interditction.head())'''


