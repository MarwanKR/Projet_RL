#test avec gym 

import json
import pandas as pd
from more_itertools import collapse
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TrainEnv(gym.Env):
    def __init__(self, file_path):
        super(TrainEnv, self).__init__()
        # Charger les données à partir du fichier JSON
        with open(file_path, 'r') as f:
            data = json.load(f)

        df_trains = pd.DataFrame(data['trains'])
        df_flat = pd.json_normalize(df_trains[0])
        df_flat['sensDepart'] = df_flat['sensDepart']*1

        self.id = df_flat['id'].tolist()
        self.number_of_trains = len(self.id)
        self.sens_depart = list(collapse(df_flat['sensDepart'].tolist()))
        self.voie_en_ligne = data["voiesEnLigne"]
        self.type_circulation = list(set(collapse(df_flat['typeCirculation'].tolist())))
        self.types_materiels = list(set(collapse(df_flat['typesMateriels'].tolist())))

        self.contraintes = pd.DataFrame(data['contraintes'])
        self.itineraire = [-1 for _ in range(self.number_of_trains)]  # Initialisation avec -1
        print(self.itineraire, "dans init")

        values = self._init_quai_interdit(data)
        self.quai_interdits = pd.DataFrame(values, columns=["voiesAQuaiInterdites", "voiesEnLigne", "typesMateriels",
                                                            "typesCirculation"])  # Dataframe des quais interdits

        self.done = False

        self.trains = df_flat  # Dataframe des trains
        # self.trains.index = self.id
        self.list_it = pd.DataFrame(data['itineraires'])  # Dataframe des itinéraires

        self.itineraire_default = np.array([int(self.list_it[(self.list_it[["sensDepart", "voieEnLigne", "voieAQuai"]]
        == self.trains.iloc[i, [1, 2, 3]]).all(1)].iloc[0, 0]) for i in range(self.number_of_trains)], dtype=np.int32)

        # Créer un dictionnaire pour l'encodage des itinéraires en indices
        self.itineraire_dict = {str(itineraire): idx for idx, itineraire in enumerate(self.list_it)}

        # Définir l'espace d'action et d'observation
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.number_of_trains),  # ID du train
            spaces.Discrete(len(self.list_it))  # ID de l'itinéraire
        ))

        # Modifié pour accepter des indices d'itinéraires (entiers)
        self.observation_space = spaces.Dict({
            "sens_depart": spaces.MultiBinary(self.number_of_trains),
            "voie_en_ligne": spaces.Box(low=np.zeros(self.number_of_trains),
                                        high=np.ones(self.number_of_trains) * (len(self.voie_en_ligne)-1),
                                        dtype=np.int32),
            "type_circulation": spaces.Box(low=np.zeros(self.number_of_trains),
                                           high=np.ones(self.number_of_trains) * (len(self.type_circulation)-1),
                                           dtype=np.int32),
            "types_materiels": spaces.Box(low=np.zeros(self.number_of_trains),
                                           high=np.ones(self.number_of_trains) * (len(self.types_materiels)-1),
                                           dtype=np.int32),
            "itineraire": spaces.Box(low=np.zeros(self.number_of_trains),
                                     high=np.ones(self.number_of_trains) * (len(self.list_it)-1),
                                     dtype=np.int32)
        })

        self.state = None

    def _init_quai_interdit(self, data):
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
        return values

    def _get_obs(self):
        if self.state is None:
            voieEnLigne = np.array([self.voie_en_ligne.index(i) for i in self.trains["voieEnLigne"]], dtype=np.int32)
            type_circulation = np.array([self.type_circulation.index(i) for i in self.trains["typeCirculation"]],
                                        dtype=np.int32)
            types_materiels = np.array([self.types_materiels.index(i[0]) for i in self.trains["typesMateriels"]],
                                       dtype=np.int32)
        else:
            voieEnLigne = self.state["voie_en_ligne"]
            type_circulation = self.state["type_circulation"]
            types_materiels = self.state["types_materiels"]
        dict = {
            "sens_depart": np.array(self.sens_depart, dtype=np.int32),
            "voie_en_ligne": voieEnLigne,
            "type_circulation": type_circulation,
            "types_materiels": types_materiels,
            "itineraire": self.itineraire
        }
        return dict

    def reset(self,seed=None, options = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.itineraire = np.copy(self.itineraire_default)
        print(self.itineraire, "dans reset")
        self.done = False
        self.state = self._get_obs()
        # print(self.state["itineraire"].shape,"dans reset")
        return self.state, {}

    def step(self, action):
        train_id, it_id = action
        self.set_itineraire(train_id, it_id)

        if self.is_it_compatible(train_id, it_id):
            reward = -1e8
        elif self.is_quaie_interdit(train_id, it_id):
            reward = -1e8
        else:
            reward = self.contraintes_itineraire(train_id, it_id)
            reward += self.any_itineraire()
        
        self.done = all(self.itineraire)

        return self.state, reward, self.done, False, {}

    def render(self, mode='human'):
        print("Current State:")
        print(self.state)

    def close(self):
        pass

    # Méthodes existantes adaptées
    def set_itineraire(self, train_id, new_itineraire):
        self.itineraire[train_id] = new_itineraire
    
    # Implémentation des vérifications et contraintes existantes...
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
            sens_depart_it = self.list_it.loc[it_id, "sensDepart"]
            voieAQuai_it = self.list_it.loc[it_id, "voieAQuai"]
            sens_depart_train = self.trains.loc[train_id, "sensDepart"]
            voieAQuai_train = self.trains.loc[train_id, "voieAQuai"]
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

    def contraintes_itineraire(self, train_id, it_id):
        c = 0
        num_train = self.trains.loc[train_id, "id"]
        index_to_check = self.contraintes[self.contraintes[[0, 1]] == [num_train, it_id]][[0, 1]].dropna().index.tolist()
        for ind in index_to_check:
            train = self.contraintes.loc[ind, 2]
            it = self.contraintes.loc[ind, 3]
            if it == self.itineraire[self.id.index(train)]:
                c += self.contraintes.loc[ind, 4]

        index_to_check = self.contraintes[self.contraintes[[2, 3]] == [num_train, it_id]][[2, 3]].dropna().index.tolist()
        for ind in index_to_check:
            train = self.contraintes.loc[ind, 0]
            it = self.contraintes.loc[ind, 1]
            if it == self.itineraire[self.id.index(train)]:
                c += self.contraintes.loc[ind, 4]
        return c

    def any_itineraire(self):
        return 1e3 * self.itineraire.count(None)


if __name__ == '__main__':
    # Exemple d'utilisation
    file_path = "instances/Asmall.json"
    trains = TrainEnv(file_path)

    from gymnasium.utils.env_checker import check_env
    check_env(trains, warn=True)
    # Afficher les données
    # dep = trains.get_sens_depart()
    # print(dep)