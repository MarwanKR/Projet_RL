import pandas as pd
import json
from more_itertools import collapse
import jax
from jax import numpy as jnp


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
        self.sens_depart = list(collapse(df_flat['sensDepart'].tolist()))
        self.voie_en_ligne = list(collapse(df_flat['voieEnLigne'].tolist()))
        self.type_circulation = list(collapse(df_flat['typeCirculation'].tolist()))
        self.types_materiels = list(collapse(df_flat['typesMateriels'].tolist()))
        self.itineraire = [None for i in range(self.number_of_trains)]
        self.done = None

        # En JAX
        # self.number_of_trains = jnp.array(len(self.id))
        # self.sens_depart = jnp.array(list(collapse(df_flat['sensDepart'].tolist())))
        # self.voie_en_ligne = jnp.array(list(collapse(df_flat['voieEnLigne'].tolist())))
        # self.type_circulation = jnp.array(list(collapse(df_flat['typeCirculation'].tolist())))
        # self.types_materiels = jnp.array(list(collapse(df_flat['typesMateriels'].tolist())))
        # self.itineraire = jnp.array([None for i in range(self.number_of_trains)])

        # Autres dataframes
        self.trains = df_flat  # Dataframe des trains
        self.trains.index = self.id
        self.list_it = pd.DataFrame(data['itineraires'])  # Dataframe des itineraires
        df_quai_interdits = pd.DataFrame(data['interdictionsQuais'])
        values = []
        if len(df_quai_interdits) > 0:
            for ind in df_quai_interdits.index:
                quais = df_quai_interdits.loc[ind, "voiesAQuaiInterdites"]
                lignes = df_quai_interdits.loc[ind, "voiesEnLigne"]
                materiels = df_quai_interdits.loc[ind, "typesMateriels"]
                circulations = df_quai_interdits.loc[ind, "typesCirculation"]
                if not lignes:
                    lignes = ['all']
                if not materiels:
                    materiels = ['all']
                if not circulations:
                    circulations = ['all']
                for quai in quais:
                    for ligne in lignes:
                        for materiel in materiels:
                            for circulation in circulations:
                                values.append([quai, ligne, materiel, circulation])
        self.quai_interdits = pd.DataFrame(values, columns=["voiesAQuaiInterdites", "voiesEnLigne", "typesMateriels",
                                                            "typesCirculation"])  # Dataframe des quais interdits
        self.contraintes = pd.DataFrame(data['contraintes'])
        # En JAX
        # self.trains = jnp.array(df_flat.to_numpy())
        # self.quai_interdits_jax = jnp.array(values)
        # self.contraintes = jnp.array(data['contraintes'])

    def reset(self):
        pass

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
            # self.data.at[index[0], 'itineraires'] = new_itineraire
            self.itineraires[index] = new_itineraire
        else:
            raise ValueError(f"Train avec l'ID {train_id} non trouvé.")

    # RESOLUTION
    def is_it_compatible(self, train_id, it_id):
        """
        Met done à true si l'itineraire qu'on propose n'est pas compatible avec le train.

        :param train_id: Train dont l'itinéraire à changer pour prendre l'itinéraire numéro it_id
        :param it_id: Vérifier la conformité de l'itinéraire avec le train (sens_depart et Voie_a_quai)
        :return : True si l'itineraire est incompatible, False (=self.done) sinon
        """
        if it_id is None:
            return self.done
        else:
            sens_depart_it = self.list_it[it_id, "sensDepart"]
            voieAQuai_it = self.list_it.loc[it_id, "voieAQuai"]
            sens_depart_train = self.trains.loc[train_id, "sensDepart"]
            voieAQuai_train = self.trains.loc[train_id, "voieAQuai"]
            if sens_depart_train == sens_depart_it and voieAQuai_train == voieAQuai_it:
                return self.done
        self.done = True
        return self.done

    def is_it_compatible_jax(self, train_id, it_id):
        """
        Si tous les Dataframe sont Jax.
        Met done à true si l'itineraire qu'on propose n'est pas compatible avec le train.

        :param train_id: Train dont l'itinéraire à changer pour prendre l'itinéraire numéro it_id
        :param it_id: Vérifier la conformité de l'itinéraire avec le train (sens_depart et Voie_a_quai)
        :return : True si l'itineraire est incompatible, False (=self.done) sinon
        """
        if it_id is None:
            return self.done
        else:
            sens_depart_it = self.list_it[it_id, 1]
            voieAQuai_it = self.list_it[it_id, 3]
            sens_depart_train = self.trains[train_id, 1]
            voieAQuai_train = self.trains[train_id, 3]
            if sens_depart_train == sens_depart_it and voieAQuai_train == voieAQuai_it:
                return self.done
        self.done = True
        return self.done

    def is_quaie_interdit(self, train_id, it_quai):
        """
        Vérifie si un train est interdit sur un quai donné en fonction des restrictions.

        :param train_id: ID du train à vérifier.
        :param quai_interdit: Liste des voies à quai interdites.
        :param ligne_interdit: Liste des voies en ligne interdites.
        :param mat_interdit: Liste des types de matériels interdits.
        :param types_interdit: Liste des types de circulations interdites.
        :return: True si le quai est interdit pour ce train, False sinon.
        """
        assert train_id in self.id
        if len(self.quai_interdits) == 0:
            return self.done

        ligne = self.trains.loc[train_id, 'voieEnLigne']
        materiel = self.trains.loc[train_id, 'typesMateriels'][0]
        circulation = self.trains.loc[train_id, 'typesCirculation']

        if (self.quai_interdits == [str(it_quai), 'all', 'all', circulation]).all(1).any():
            self.done = True
        elif (self.quai_interdits == [str(it_quai), 'all', materiel, 'all']).all(1).any():
            self.done = True
        elif (self.quai_interdits == [str(it_quai), 'all', materiel, circulation]).all(1).any():
            self.done = True
        elif (self.quai_interdits == [str(it_quai), ligne, 'all', 'all']).all(1).any():
            self.done = True
        elif (self.quai_interdits == [str(it_quai), ligne, 'all', circulation]).all(1).any():
            self.done = True
        elif (self.quai_interdits == [str(it_quai), ligne, materiel, 'all']).all(1).any():
            self.done = True
        elif (self.quai_interdits == [str(it_quai), ligne, materiel, circulation]).all(1).any():
            self.done = True

        return self.done

    def is_quaie_interdit_jax(self, train_id, it_quai):
        """
        Si tous les DataFrames sont en JAX.
        Vérifie si un train est interdit sur un quai donné en fonction des restrictions.

        :param train_id: ID du train à vérifier.
        :param quai_interdit: Liste des voies à quai interdites.
        :param ligne_interdit: Liste des voies en ligne interdites.
        :param mat_interdit: Liste des types de matériels interdits.
        :param types_interdit: Liste des types de circulations interdites.
        :return: True si le quai est interdit pour ce train, False sinon.
        """
        assert train_id in self.id
        if len(self.quai_interdits) == 0:
            return self.done

        ligne = self.trains[train_id, 2]
        materiel = self.trains[train_id, 6][0]
        circulation = self.trains[train_id, 4]

        c1 = (self.quai_interdits[:, 0] == str(it_quai))
        c2 = (self.quai_interdits[:, 1] == ligne) + (self.quai_interdits[:, 1] == 'all')
        c3 = (self.quai_interdits[:, 2] == materiel) + (self.quai_interdits[:, 2] == 'all')
        c4 = (self.quai_interdits[:, 3] == circulation) + (self.quai_interdits[:, 3] == 'all')
        if (c1 * c2 * c3 * c4).any():
            self.done = True

        return self.done

    def contraintes_itineraire(self, train_id, it_id):
        c = 0
        index_to_check = self.contraintes[self.contraintes[[0, 1]] == [train_id, it_id]][[0, 1]].dropna().index.tolist()
        for ind in index_to_check:
            train = self.contraintes.loc[ind, 2]
            it = self.contraintes.loc[ind, 3]
            if it == self.itineraire[self.id.index(train)]:
                c += self.contraintes.loc[ind, 4]

        index_to_check = self.contraintes[self.contraintes[[2, 3]] == [train_id, it_id]][[2, 3]].dropna().index.tolist()
        for ind in index_to_check:
            train = self.contraintes.loc[ind, 0]
            it = self.contraintes.loc[ind, 1]
            if it == self.itineraire[self.id.index(train)]:
                c += self.contraintes.loc[ind, 4]
        return c

    def contraintes_itineraire_jax(self, train_id, it_id):
        c = 0
        index_to_check = (self.contraintes[:, [0, 1]] == [train_id, it_id]).all(1)
        for i, ind in enumerate(index_to_check):  # TODO : Supprimer la boucle for
            if ind:
                train = self.contraintes[i, 2]
                it = self.contraintes.loc[i, 3]
                if it == self.itineraire[self.id.index(train)]:
                    c += self.contraintes[i, 4]

        index_to_check = (self.contraintes[:, [2, 3]] == [train_id, it_id]).all(1)
        for i, ind in enumerate(index_to_check):
            if ind:
                train = self.contraintes[i, 0]
                it = self.contraintes.loc[i, 1]
                if it == self.itineraire[self.id.index(train)]:
                    c += self.contraintes[i, 4]
        return c

    def any_itineraire(self, c=1e3):
        """
        Retourne le coût des non attribution des itinéraires.
        :param c: coût d'un itinéraire non attribué
        :return:
        """
        return c * self.itineraire.count(None)

    # ENVIRONNEMENT
    def step(self, action):
        """
        pour une action donnée (train, itineraire), mettera à jour le state, done et calculera le coût de l'action.

        :param action: tuple (train_id, it_id)
        :return:
        """
        self.set_itineraire(train_id=action[0], new_itineraire=action[1])
        # Compatibilité
        if self.is_it_compatible(train_id=action[0], it_id=action[1]):
            reward = - 1e8
        elif self.is_quaie_interdit(train_id=action[0], it_quai=action[1]):
            reward = - 1e8
        else:
            reward = self.contraintes_itineraire(train_id=action[0], it_id=action[1])  # On décide de prendre que la
            # contrainte du nouvel itineraire ? Sinon construire la fonction des calcul des contraintes.
            reward += self.any_itineraire()  # Coût des quais non-attribués

        return reward, self.done


# Exemple d'utilisation
file_path = "instances/inst_PMP.json"
trains = Trains(file_path)

# Afficher les données
dep = trains.get_sens_depart()
print(dep)
