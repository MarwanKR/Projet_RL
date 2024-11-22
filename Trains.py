import pandas as pd
import json

class Trains:
    def __init__(self, file_path):
        # Charger les données à partir du fichier JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        df_trains = pd.DataFrame(data['trains'])
        df_flat = pd.json_normalize(df_trains[0])
        # Créer des attributs correspondant à chaque colonne
        self.id = df_flat['id'].tolist()
        self.number_of_trains = len(self.id)
        self.sens_depart = df_flat['sensDepart'].tolist()
        self.voie_en_ligne = df_flat['voieEnLigne'].tolist()
        self.type_circulation = df_flat['typeCirculation'].tolist()
        self.types_materiels = df_flat['typesMateriels'].tolist()
        self.itineraire = [None for i in range(self.number_of_trains)]
        self.done = None
    
    # Getters
    def get_ids(self):
        """Retourne la liste des IDs des trains."""
        return self.id

    def get_sens_depart(self):
        """Retourne la liste des directions des trains (True/False)."""
        return self.sens_depart

    def get_voie_en_ligne(self):
        """Retourne la liste des voies en ligne des trains."""
        return self.voie_en_ligne

    def get_type_circulation(self):
        """Retourne la liste des types de circulation."""
        return self.type_circulation

    def get_types_materiels(self):
        """Retourne la liste des types de matériels."""
        return self.types_materiels
    
    # Exemple de méthode personnalisée
    def get_trains_by_type(self, type_circulation):
        """
        Filtre les trains par type de circulation.
        :param type_circulation: Type de circulation à filtrer (ex: 'FRET', 'TGV').
        :return: Un DataFrame des trains correspondants.
        """
        return self.data[self.data['typeCirculation'] == type_circulation]
    
    def dataframe_to_json(self, file_path=None, orient="records", indent=4):
        """
        Convertit un DataFrame en JSON et, optionnellement, le sauvegarde dans un fichier.
        
        :param df: Le DataFrame pandas à convertir.
        :param file_path: Chemin du fichier où sauvegarder le JSON (optionnel).
        :param orient: Orientation des données dans le JSON.
                    Exemples : 'records', 'split', 'index', 'columns', 'values'.
        :param indent: Indentation pour la lisibilité (par défaut 4).
        :return: Le JSON sous forme de chaîne de caractères.
        """
        # Convertir le DataFrame en JSON
        json_data = self.data.to_json(orient=orient, indent=indent)
        
        # Sauvegarder dans un fichier si un chemin est spécifié
        if file_path:
            with open(file_path, "w") as json_file:
                json_file.write(json_data)
        
        return json_data
    
    def set_itineraire(self, train_id, new_itineraire):
        """
        Met à jour l'itinéraire d'un train donné par son ID.
        
        :param train_id: L'ID du train à mettre à jour.
        :param new_itineraire: La nouvelle valeur de l'itinéraire à attribuer au train.
        """
        # Trouver l'indice du train avec l'ID correspondant
        index = self.data[self.data['id'] == train_id].index

        # Si l'ID existe, mettre à jour l'itinéraire
        if not index.empty:
            #self.data.at[index[0], 'itineraires'] = new_itineraire
            self.itineraires[index] = new_itineraire
        else:
            raise ValueError(f"Train avec l'ID {train_id} non trouvé.")
        
    def is_it_compatible(self, train_id, it_sens, it_ligne):
        '''
        Met done à true si l'itineraire qu'on propose n'est pas compatible avec le train
        '''
        index = self.data[self.data['id'] == train_id].index

        if not index.empty:
            if self.sens_depart[index] != it_sens or self.voie_en_ligne != it_ligne:
                self.done = True
        else:  
            raise ValueError(f"Train avec l'ID {train_id} non trouvé.")
        
        return self.done
    
    def is_quaie_interdit(self, train_id, it_quai, it_ligne, quai_interdit, ligne_interdit, mat_interdit, types_interdit):
        """
        Vérifie si un train est interdit sur un quai donné en fonction des restrictions.

        :param train_id: ID du train à vérifier.
        :param quai_interdit: Liste des voies à quai interdites.
        :param ligne_interdit: Liste des voies en ligne interdites.
        :param mat_interdit: Liste des types de matériels interdits.
        :param types_interdit: Liste des types de circulations interdites.
        :return: True si le quai est interdit pour ce train, False sinon.
        """
        index = self.data[self.data['id'] == train_id].index

        if not index.empty:
            if quai_interdit != [] and it_quai not in quai_interdit:
                return self.done
            elif ligne_interdit != [] and it_ligne not in ligne_interdit:
                return self.done
            elif mat_interdit != [] and not set(self.get_types_materiels[index]).intersection(set(mat_interdit)):
                return self.done
            elif types_interdit != [] and self.get_type_circulation[index] not in types_interdit:
                return self.done
            else:
                self.done = True
                return self.done
        else:
            raise ValueError(f"Train avec l'ID {train_id} non trouvé.")



# Exemple d'utilisation
file_path = "/Users/elie/Desktop/Polytechnique/RL/Projet/instances/Asmall.json"
trains = Trains(file_path)

# Afficher les données
dep = trains.get_sens_depart()
print(dep)
