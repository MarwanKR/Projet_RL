import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time 

class Environnment : 
    def __init__(self,file_path,learning_rate = 0.001):
        with open(file_path, 'r') as f:
            data = json.load(f)

        #Dataframes
        df_trains = pd.DataFrame(data['trains'])
        df_flat = pd.json_normalize(df_trains[0])
        df_flat['sensDepart'] = df_flat['sensDepart']*1

        self.trains = df_flat  # Dataframe des trains
        self.id = df_flat['id'].tolist()
        self.number_of_trains = len(self.id)
        self.list_it = pd.DataFrame(data['itineraires'])  # Dataframe des itinéraires
        self.contraintes = pd.DataFrame(data['contraintes'])
        values = self._init_quai_interdit(data)
        self.quai_interdits = pd.DataFrame(values, columns=["voiesAQuaiInterdites", "voiesEnLigne", "typesMateriels","typesCirculation"])  # Dataframe des quais interdits
        #parameters and hyperparameters
        self.learning_rate =learning_rate
        self.action_spaces = {}
        self.done = False
        for _, train in self.trains.iterrows():  # Iterate over train rows
            train_action_space = []
            for _, itinerary in self.list_it.iterrows():  # Iterate over itinerary rows
                # Check conditions for adding an itinerary to the train's action space
                if itinerary["sensDepart"] * 1 == train["sensDepart"] and itinerary["voieEnLigne"] == train["voieEnLigne"]:
                    train_action_space.append(itinerary)

            self.action_spaces[train["id"]] = train_action_space

        self.state_space = self.trains["id"]
        self.assigned_itineraries = {}
        self.current_state_index = 0
        self.current_state = self.state_space[0] #id du train actuel
        print("current state init",self.current_state)
        print(self.state_space[1],"deuxième state")
        self.past_state = None #id du train traité précédemment
        self.Q_values = {}
        self.number_of_visits = {}
        for state in self.state_space :
            self.assigned_itineraries[state] = len(self.list_it) 
            for action in self.action_spaces[state]:
                self.Q_values[(state,action["id"])]= 0
                self.number_of_visits[(state,action["id"])] = 0

        self.cost = -self.number_of_trains*500
        print("assigned itineraries init",self.assigned_itineraries)

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

    def print_env(self):
        print(self.trains.head())
        print("the state space (les id des trains)",print(self.state_space))

    def select_action(self,state):
        action_space = self.action_spaces[state]
        epsilon = np.random.random()
        if epsilon <= 0.05:
            return np.random.choice([action["id"] for action in action_space])
        # Vectorized max selection
        q_values = np.array([self.Q_values[(state, action["id"])] for action in action_space])
        max_index = np.argmax(q_values)
        return action_space[max_index]["id"]
    

    def set_itinerary(self,train_id,it_id):
        self.assigned_itineraries[train_id] = it_id

    def contraintes_itineraire(self, train_id, it_id):

        if it_id == len(self.list_it):
            return 500

        # Filter rows where train_id and it_id are relevant
        mask = (
            ((self.contraintes[0] == train_id) & (self.contraintes[1] == it_id)) |
            ((self.contraintes[2] == train_id) & (self.contraintes[3] == it_id))
        )

        filtered_contraintes = self.contraintes[mask]
        
        # Calculate the penalty cost
        c = filtered_contraintes[
            (filtered_contraintes[0] == train_id) &
            (filtered_contraintes[1] == it_id) &
            (filtered_contraintes[2].map(self.assigned_itineraries.get) == filtered_contraintes[3])
        ][4].sum()

        c += filtered_contraintes[
            (filtered_contraintes[2] == train_id) &
            (filtered_contraintes[3] == it_id) &
            (filtered_contraintes[0].map(self.assigned_itineraries.get) == filtered_contraintes[1])
        ][4].sum()
        return c/100

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
        
        ligne = self.trains.loc[self.current_state_index, 'voieEnLigne']
        materiel = self.trains.loc[self.current_state_index, 'typesMateriels'][0]
        circulation = self.trains.loc[self.current_state_index, 'typeCirculation']

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

    def step(self ):
        current_state = self.current_state
        action_id = self.select_action(current_state)
        self.set_itinerary(current_state,action_id)
        
        reward = -self.contraintes_itineraire(current_state,action_id)/100
        
        if self.is_quaie_interdit(current_state,action_id):
            reward = -1000
            self.done = True
        self.cost+=reward/100
        self.Q_values[(current_state,action_id)]  = (self.Q_values[(current_state,action_id)]*self.number_of_visits[(current_state,action_id)] +reward)/(self.number_of_visits[(current_state,action_id)]+1)
        self.number_of_visits[(current_state,action_id)]+=1
        self.current_state_index+=1
        if self.current_state_index==self.number_of_trains : 
            self.done = True
        else : 
            next_state = self.trains.loc[self.current_state_index, "id"]
            self.current_state = next_state
        
    
    def get_total_cost_of_config(self):
        cost = 0
        for train_id in self.trains["id"]:
            it_id = self.assigned_itineraries[train_id]
            cost+=self.contraintes_itineraire(train_id,it_id)/100
        return -cost
    def reset(self):
        self.done = False 
        self.current_state_index = 0
        self.current_state = self.trains["id"][0]
        self.assigned_itineraries = {}
        self.cost = 0
        for state in self.state_space :
            self.assigned_itineraries[state] = len(self.list_it)



    def train_model(self,number_of_episodes):
        list_of_cost = []
        for i in range(number_of_episodes):
            while not self.done:
                self.step()
            print("episode")
            list_of_cost.append(self.get_total_cost_of_config())
            if i%100==0 :
                print(self.assigned_itineraries) 
            self.reset()
          
        return list_of_cost
            


    






if __name__ == '__main__':
    env = Environnment('instances/inst_NS.json')
    env.print_env()

    results = env.train_model(201)
    plt.plot(results)
    plt.show()