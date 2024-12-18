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

        # Action-space
        self.itineraire = [len(self.list_it) for _ in range(self.number_of_trains)]  # Initialisation avec None
        print(self.itineraire, "dans init")


        # # Construction de l'itinéaire par défault fournit par la SNCF
        # array = []
        # for i in range(self.number_of_trains):
        #     try:
        #         ind = int(self.list_it[(self.list_it[["sensDepart", "voieEnLigne", "voieAQuai"]] == self.trains.iloc[i, [1, 2, 3]]).all(1)].iloc[0, 0])
        #     except IndexError:  # L'itinéraire attribué est inéxistant/incompatible
        #         ind = len(self.list_it)
        #     array.append(ind)
        # self.itineraire_default = np.array(array, dtype=np.int32)

        # Créer un dictionnaire pour l'encodage des itinéraires en indices

        # Définir l'espace d'action et d'observation
        self.action_space = spaces.Discrete(len(self.list_it)+1)
        self.observation_space = spaces.Discrete(self.number_of_trains)

        self.state = None

        self.render_mode = 'human'
        self.last_it = self.itineraire
        self.cost = 0
        self.first_cost = self._get_info(reset=True)['cost_config']

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
        num_train = self.current_id[self.current_step]
        obs = self.id.index(num_train)
        return obs

    def _get_info(self, train=None, it=None, reset=False):
        if reset:
            cost = 0
            if self.done:
                return {'cost_config': -30, 'itineraire': self.itineraire}
            for train in range(self.number_of_trains):
                cost -= self.contraintes_itineraire(train, self.itineraire[train])
            self.cost = cost
        else:
            last_cost = self.cost
            self.cost = last_cost + self.contraintes_itineraire(train, self.last_it[train])
            self.cost -= self.contraintes_itineraire(train, it)
            if self.cost == 0:
                print(f'itinéraire :{self.itineraire}')
        return {'cost_config': self.cost, 'itineraire': self.itineraire}

    def reset(self, seed=None, options=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.current_step = 0
        np.random.shuffle(self.current_id)
        self.itineraire = [len(self.list_it) for _ in range(self.number_of_trains)]  # Itinéraire de la SNCF par défaut
        self.last_it = np.copy(self.itineraire)
        
        self.done = False
        self.state = self._get_obs()
        self.cost = self.first_cost
        # print(self.state["itineraire"].shape,"dans reset")
        return self.state, {}

    def step(self, action):
        train_id = self._get_obs()
        it_id = action
        self.set_itineraire(train_id, it_id)

        if not self.is_it_incompatible(train_id, it_id) and not self.is_quaie_interdit(train_id, it_id):
            reward = -self.contraintes_itineraire(train_id, it_id)/ 100.0
            info = self._get_info(train_id, it_id)

        else:
            reward = -1000
            info = {'cost_config': 100, 'itineraire': self.itineraire}

        if self.current_step == self.number_of_trains-1:
            self.done = True
        else:
            self.current_step += 1
        return self._get_obs(), reward, self.done, False, info

    def render(self, mode='human'):
        # print("Current Itineraire:")
        # print(self.itineraire)
        cost = 0
        for train in range(self.number_of_trains):
            if self.done:
                return -30
            cost -= self.contraintes_itineraire(train, self.itineraire[train])
        return cost


    def close(self):
        pass

    # Méthodes existantes adaptées
    def set_itineraire(self, train_id, new_itineraire):
        self.last_it = np.copy(self.itineraire)
        self.itineraire[train_id] = new_itineraire
    
    # Implémentation des vérifications et contraintes existantes...
    def is_it_incompatible(self, train_id, it_id):
        """
        Met done à true si l'itineraire qu'on propose n'est pas compatible avec le train.

        :param train_id: Train dont l'itinéraire à changer pour prendre l'itinéraire numéro it_id
        :param it_id: Vérifier la conformité de l'itinéraire avec le train (sens_depart et Voie_a_quai)
        :return : True si l'itineraire est incompatible, False (=self.done) sinon
        """
        if it_id == len(self.list_it):
            return self.done
        else:
            sens_depart_it = self.list_it.loc[it_id, "sensDepart"]
            voieEnLigne_it = self.list_it.loc[it_id, "voieEnLigne"]
            sens_depart_train = self.trains.loc[train_id, "sensDepart"]
            voieEnLigne_train = self.trains.loc[train_id, "voieEnLigne"]
            if sens_depart_train == sens_depart_it and voieEnLigne_train == voieEnLigne_it:
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
        if len(self.quai_interdits) == 0 or it_quai == len(self.list_it):
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
        if it_id == len(self.list_it):
            return 50000
        c = 0
        num_train = self.trains.loc[train_id, "id"]
        index_to_check = self.contraintes[self.contraintes[[0, 1]] == [num_train, it_id]][[0, 1]].dropna().index.tolist()
        for ind in index_to_check:
            train = self.contraintes.loc[ind, 2]
            it = self.contraintes.loc[ind, 3]
            if train in self.id and it == self.itineraire[self.id.index(train)]:
                c += self.contraintes.loc[ind, 4]

        index_to_check = self.contraintes[self.contraintes[[2, 3]] == [num_train, it_id]][[2, 3]].dropna().index.tolist()
        for ind in index_to_check:
            train = self.contraintes.loc[ind, 0]
            it = self.contraintes.loc[ind, 1]
            try:
                if it == self.itineraire[self.id.index(train)]:
                    c += self.contraintes.loc[ind, 4]
            except ValueError:
                pass
        return c


if __name__ == '__main__':
    # Example usage
    random_seed = 10
    file_path = "instances/inst_A.json"
    env = TrainEnv(file_path)

    # check_env(env)
    # Display data
    dep = env.sens_depart
    print(dep, "sens des départs")

    # Step 3: Vectorize the environment
    vec_env = DummyVecEnv([lambda: env])

    # Step 4: Create and train the DQN model
    model = DQN("MlpPolicy", vec_env, verbose=0, learning_rate=1e-3, buffer_size=800, seed=random_seed)
    number_of_episodes = 2000
    max_number_of_steps = env.number_of_trains
    # Tracking cumulative rewards
    episode_rewards = []  # List to store total reward per episode
    episode_rewards2 = []
    cumulative_reward = 0  # Cumulative reward for the current episode
    cumulative_reward2 = 0
    track_number_of_steps = []
    Itineraires = []
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
                cumulative_reward = reward + cumulative_reward*0.95  # Accumulate rewards

            Itineraires.append(info[0]['itineraire'])
            track_number_of_steps.append(number_of_steps)
            episode_rewards.append(cumulative_reward)  # Log reward for this episode
            episode_rewards2.append(info[0]['cost_config'])
            pbar.update(1)

    # Plot the evolution of expected return
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label="Cumulative Reward")
    plt.plot(episode_rewards2, label="Total cost")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Evolution of Expected Return During Training")
    plt.legend()
    plt.figure(figsize=(10, 6))
    plt.plot(track_number_of_steps, label="number of steps")
    plt.xlabel("Episode")
    plt.ylabel("number of steps")
    plt.title("Evolution of number of steps before done")
    plt.legend()
    plt.grid()
    plt.figure(figsize=(10, 6))
    plt.plot(Itineraires, 'o', label=[f'train {i}' for i in range(env.number_of_trains)])
    plt.xlabel("Episode")
    plt.ylabel("Itineraire")
    plt.title("Evolution des itinéraires finaux")
    plt.legend()
    plt.grid()
    plt.show()