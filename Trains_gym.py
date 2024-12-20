#test avec gym
import copy
import json
import pandas as pd
from more_itertools import collapse
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN,PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm 
import matplotlib.pyplot as plt

class TrainEnv(gym.Env):
    def __init__(self, file_path):
        super(TrainEnv, self).__init__()
        # Charger les données à partir du fichier JSON
        with open(file_path, 'r') as f:
            data = json.load(f)

        df_trains = pd.DataFrame(data['trains'])
        df_flat = pd.json_normalize(df_trains[0])
        df_flat['sensDepart'] = df_flat['sensDepart']*1

        # Dataframes
        self.trains = df_flat  # Dataframe des trains
        self.list_it = pd.DataFrame(data['itineraires'])  # Dataframe des itinéraires
        self.list_it['sensDepart'] = self.list_it['sensDepart']*1
        self.contraintes = pd.DataFrame(data['contraintes'])
        values = self._init_quai_interdit(data)
        self.quai_interdits = pd.DataFrame(values, columns=["voiesAQuaiInterdites", "voiesEnLigne", "typesMateriels",
                                                            "typesCirculation"])  # Dataframe des quais interdits

        # State (id des trains)
        self.id = df_flat['id'].tolist()
        self.current_id = np.copy(self.id)
        self.current_step = 0

        # Paramètres
        self.number_of_trains = len(self.id)
        self.sens_depart = list(collapse(df_flat['sensDepart'].tolist()))
        self.voie_en_ligne = data["voiesEnLigne"]
        self.type_circulation = list(set(collapse(df_flat['typeCirculation'].tolist())))
        self.types_materiels = list(set(collapse(df_flat['typesMateriels'].tolist())))
        self.done = False

        # Construction de l'itinéaire par défault fournit par la SNCF
        array = []
        for i in range(self.number_of_trains):
            try:
                ind = int(self.list_it[(self.list_it[["sensDepart", "voieEnLigne", "voieAQuai"]] == self.trains.iloc[i, [1, 2, 3]]).all(1)].iloc[0, 0])
            except IndexError:  # L'itinéraire attribué est inéxistant/incompatible
                ind = len(self.list_it)
            array.append(ind)
        self.itineraire_default = np.array(array, dtype=np.int32)

        # # Construction test
        # new_list_it = pd.DataFrame(self.list_it.loc[self.itineraire_default,:].values, columns=self.list_it.columns)
        # new_list_it.loc[len(new_list_it), :] = [1, True, 'ZZ', 'ZZ']
        # new_list_it.loc[len(new_list_it), :] = self.list_it.loc[317, :]
        # self.list_it = new_list_it.copy()

        # Construction dictionnaire des itinéraires compatibles
        self.dict_it = {}
        self.len_compatible_it = []
        with tqdm(total=self.number_of_trains, desc="Initialize Progress") as pbar:
            for train_id in range(self.number_of_trains):
                compatible = self.list_it[(self.list_it[['sensDepart', 'voieEnLigne']] == self.trains.loc[
                    train_id, ['sensDepart', 'voieEnLigne']].to_list()).all(1)].index.tolist()
                correct = []
                while compatible:
                    it_id = compatible.pop(0)
                    if not self.is_quaie_interdit(train_id, it_id):
                        correct.append(it_id)
                assert len(correct) > 0, f"Le train {train_id} n'admet aucun itinéraire compatible et autorisé"
                self.dict_it[train_id] = correct
                self.len_compatible_it.append(len(correct))
                pbar.update(1)


        # Action-space
        # self.max_it = max(self.len_compatible_it)
        self.itineraire = {}
        for train_num in self.id:
            self.itineraire[train_num] = len(self.list_it)  # Initialisation avec None
        print(self.itineraire, "dans init")

        # Définir l'espace d'action et d'observation
        self.action_space = spaces.Discrete(np.max(self.len_compatible_it))
        self.observation_space = spaces.Discrete(self.number_of_trains+1)

        self.state = None

        self.render_mode = 'human'
        self.last_it = self.itineraire.copy()
        self.cost = 0
        self.first_cost = 0  # self._get_info(reset=True)['cost_config']

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
        if not self.done:
            num_train = self.current_id[self.current_step]
            obs = self.id.index(num_train)
        else:
            obs = self.current_step
        return obs

    def _get_info(self, do=False):
        info = {'itineraire': list(self.itineraire.values())}
        if do:
            cost = 0
            for train in range(self.number_of_trains):
                cost -= self.contraintes_itineraire(train, self.itineraire[self.id[train]])
            self.cost = cost
            info = {'cost_config': self.cost, 'itineraire': list(self.itineraire.values())}
        return info

    def reset(self, seed=None, options=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.current_step = 0
        # np.random.shuffle(self.current_id)
        self.itineraire = self.last_it.copy()
        
        self.done = False
        self.state = self._get_obs()
        self.cost = 0  # self.first_cost
        # print(self.state["itineraire"].shape,"dans reset")
        return self.state, {}

    def step(self, action):
        train_id = self._get_obs()
        it_id = self.dict_it[train_id][action % self.len_compatible_it[train_id]]
        self.set_itineraire(train_id, it_id)
        reward = -self.contraintes_itineraire(self.id[train_id], it_id)
        self.cost += reward
        info = {}
        # print(self.current_step)
        # if not self.is_it_incompatible(train_id, it_id) and not self.is_quaie_interdit(train_id, it_id):


        # else:
        #     self.done = True
        #     reward = -20000
        #     info = {'cost_config': 100, 'itineraire': self.itineraire}

        self.current_step += 1
        if self.current_step == self.number_of_trains:
            self.done = True
            info = {'cost_config': self.cost, 'itineraire': list(self.itineraire.values())}  # self._get_info(do=True)
            # if info['cost_config']*reward == 0:
            #     print(info['cost_config'])
            #     pass
        return self._get_obs(), reward, self.done, False, info


    def close(self):
        pass

    # Méthodes existantes adaptées
    def set_itineraire(self, train_id, new_itineraire):
        self.itineraire[self.id[train_id]] = new_itineraire
    
    # Implémentation des vérifications et contraintes existantes...
    def is_it_incompatible(self, train_id, it_id):
        """
        Met done à true si l'itineraire qu'on propose n'est pas compatible avec le train.

        :param train_id: Train dont l'itinéraire à changer pour prendre l'itinéraire numéro it_id
        :param it_id: Vérifier la conformité de l'itinéraire avec le train (sens_depart et Voie_a_quai)
        :return : True si l'itineraire est incompatible, False (=self.done) sinon
        """
        if it_id == len(self.list_it):
            return False
        else:
            sens_depart_it = self.list_it.loc[it_id, "sensDepart"]
            voieEnLigne_it = self.list_it.loc[it_id, "voieEnLigne"]
            sens_depart_train = self.trains.loc[train_id, "sensDepart"]
            voieEnLigne_train = self.trains.loc[train_id, "voieEnLigne"]
            if sens_depart_train == sens_depart_it and voieEnLigne_train == voieEnLigne_it:
                return False
        # self.done = True
        return True

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
        if len(self.quai_interdits) == 0 or it_quai == len(self.list_it):
            return False

        ligne = self.trains.loc[train_id, 'voieEnLigne']
        materiel = self.trains.loc[train_id, 'typesMateriels'][0]
        circulation = self.trains.loc[train_id, 'typeCirculation']

        if (self.quai_interdits == [str(it_quai), 'all', 'all', circulation]).all(1).any():
            return True
        elif (self.quai_interdits == [str(it_quai), 'all', materiel, 'all']).all(1).any():
            return True
        elif (self.quai_interdits == [str(it_quai), 'all', materiel, circulation]).all(1).any():
            return True
        elif (self.quai_interdits == [str(it_quai), ligne, 'all', 'all']).all(1).any():
            return True
        elif (self.quai_interdits == [str(it_quai), ligne, 'all', circulation]).all(1).any():
            return True
        elif (self.quai_interdits == [str(it_quai), ligne, materiel, 'all']).all(1).any():
            return True
        elif (self.quai_interdits == [str(it_quai), ligne, materiel, circulation]).all(1).any():
            return True

        return False

    def contraintes_itineraire(self, train_num, it_id):
        if it_id == len(self.list_it):
            return 20

        # Filter rows where train_id and it_id are relevant
        mask = (
                ((self.contraintes[0] == train_num) & (self.contraintes[1] == it_id)) |
                ((self.contraintes[2] == train_num) & (self.contraintes[3] == it_id))
        )

        filtered_contraintes = self.contraintes[mask]
        # Calculate the penalty cost
        c = filtered_contraintes[
            (filtered_contraintes[0] == train_num) &
            (filtered_contraintes[1] == it_id) &
            (filtered_contraintes[2].map(self.itineraire.get) == filtered_contraintes[3])
            ][4].sum()

        c += filtered_contraintes[
            (filtered_contraintes[2] == train_num) &
            (filtered_contraintes[3] == it_id) &
            (filtered_contraintes[0].map(self.itineraire.get) == filtered_contraintes[1])
            ][4].sum()

        return c/1000


if __name__ == '__main__':
    # Example usage
    random_seed = 10
    file_path_list = ["Asmall", "inst_A", "inst_NS", "inst_PMP"]
    for file_path in file_path_list:
        print("Begin Initialisation...")
        env = TrainEnv("instances/"+file_path+".json")

        # check_env(env)
        # Display data
        dep = env.sens_depart
        print(dep, "sens des départs")

        # Step 3: Vectorize the environment
        vec_env = DummyVecEnv([lambda: env])

        # Step 4: Create and train the DQN model
        model = DQN("MlpPolicy", vec_env, verbose=0, learning_rate=1e-3, buffer_size=800, target_update_interval=100)
        number_of_episodes = 200
        max_number_of_steps = env.number_of_trains
        # Tracking cumulative rewards
        episode_rewards = []  # List to store total reward per episode
        episode_rewards2 = []
        cumulative_reward = 0  # Cumulative reward for the current episode
        cumulative_reward2 = 0
        track_number_of_steps = []
        Itineraires = []
        max = -1e5  # Best cost
        # Progress bar
        with tqdm(total=number_of_episodes, desc="Training Progress") as pbar:
            for _ in range(number_of_episodes):
                # Simulate environment to track cumulative rewards
                obs = vec_env.reset()  # Reset the environment
                done = False
                cumulative_reward = 0  # Reset reward for new episode
                cumulative_reward2 = 0
                number_of_steps = 0
                while not done and number_of_steps<max_number_of_steps:
                    model.learn(total_timesteps=1, reset_num_timesteps=False)
                    number_of_steps += 1
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = vec_env.step(action)
                    # cumulative_reward2 += info[0]['cost_config']  # Accumulate rewards
                    cumulative_reward = reward + cumulative_reward  # Accumulate rewards
                if info[0]['cost_config'] > max:
                    max = info[0]['cost_config']
                    itinary = info[0]['itineraire']
                episode_rewards.append(cumulative_reward)  # Log reward for this episode
                episode_rewards2.append(info[0]['cost_config'])
                pbar.update(1)
        print(file_path)
        print(max)
        print(itinary)
        # Plot the evolution of expected return
        plt.figure(figsize=(10, 6))
        # plt.plot(episode_rewards, label="Cumulative Reward")
        plt.plot(episode_rewards2, label=f"instance {file_path}")
        plt.xlabel("Episode")
        plt.ylabel("Total cost of the configuration")
        plt.title("Evolution of the total cost during training")
        plt.legend()

        # Enregistrer la figure au format PNG
        plt.savefig(f'figure_DQN_{file_path}.png', dpi=300, bbox_inches='tight')  # Résolution 300 DPI

    plt.show()