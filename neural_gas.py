#import optuna 
import os 
import networkx as nx 
import numpy as np
import random
from numpy.lib.stride_tricks import sliding_window_view

from scipy.spatial.distance import cdist
from scipy.stats import rankdata, norm
from tqdm import tqdm 
import pickle

class GrowingNeuralGas():
    
    def __init__(self,
                 lambda_constante,
                 learning_cible,
                 learning_neighbors,
                 error_damping_factor,
                 threshold_age_edge,
                 threshold_error = 100_000,
                 nb_iteration = 500_000,
                 fraction_node = 0.01,
                 nb_node = 500,
                 tolerance=50
                 ) -> None:
        
        self.initialize_graph()

        self.node_max = 1

        # Hyper parameter
        self.lambda_constante = lambda_constante
        self.learning_cible = learning_cible
        self.learning_neighbors = learning_neighbors
        self.error_damping_factor = error_damping_factor
        self.threshold_age_edge = threshold_age_edge

        # Early Stopping parameters
        self.iteration_criterion = nb_iteration
        self.fraction_node = fraction_node
        self.nb_node = nb_node
        self.error_thresh = threshold_error
        self.error_memory = 0
        self.prev_error = 0
        self.best_bic = 1e10
        self.tolerance = tolerance
        
    def initialize_graph(self):
        self.G = nx.Graph()
        self.G.add_nodes_from([0,1])
        nx.set_node_attributes(self.G, {node : 0 for node in self.G.nodes}, name="error")
        #nx.set_node_attributes(self.G, {node : 0 for node in self.G.nodes}, name="SE")
        nx.set_node_attributes(self.G, {node : 0 for node in self.G.nodes}, name="n_memb")
    
    def fit(self, X_train, X_test):
        self.position_matrix = np.random.uniform(-1,1,size=(2,X_train.shape[1]))
        
        iterate = 1
        pbar = tqdm(total=10_000)
        while iterate != 0:

            # 0 - Get the data
            ind_X = random.choice([i for i in range(X_train.shape[0])])
            X = np.reshape(X_train[ind_X,:],(1,-1))

            #print(X.shape, self.position_matrix.shape)
            # 1 - On calcul les distances et on classe les noeuds du plus proche au moins proche
            distances = np.reshape(cdist(X,self.position_matrix),(-1))
            rank_distance = rankdata(distances,method="ordinal")
            #print(distances, rank_distance)
            # 2 - On identifie les deux noeuds qui sont connectés
            s1 = np.array(self.G.nodes)[rank_distance==1][0]
            s2 = np.array(self.G.nodes)[rank_distance==2][0]
            #print(s1)
            # 3 - On update l'age des liens
            self.age_update(s1, [neighbor for neighbor in self.G.neighbors(s1)])

            # 4 - On update le taux d'erreur du noeud
            self.error_update(s1, distances[rank_distance==1])

            # 5 - On déplace le noeud et ses voisins
            self.position_update(s1,[neighbor for neighbor in self.G.neighbors(s1)], X)

            # 6 - On met à jours les liens et retire les noeuds non connectés
            self.update_link(s1,s2)

            # 7 - On ajoute un noeud si il le faut
            if iterate%self.lambda_constante == 0:
                self.add_neuron()

            # 8 - Error update
            self.error_damping()

            # 9 - Early_stopping criterion
            
            if iterate == self.iteration_criterion:
                break
            elif len(list(self.G.nodes)) > 0.01*X_train.shape[0]:
                break 
            elif len(list(self.G.nodes)) > self.nb_node:
                break
            elif iterate % 20 == 0:
                #ind_X = random.choices([i for i in range(X_train.shape[0])],k=200)
                #X = X_train[ind_X,:]
                distances = cdist(X_test,self.position_matrix)
                #print(distances.shape)
                rangs = rankdata(distances,method="ordinal",axis=1)
                bic = self.BIC(distances,rangs)
                print(bic)
                if self.error_stopping(bic):
                    break
            else:
                ok = 1
            iterate+=1
            pbar.update(1)
            
    def age_update(self,s1, neighbors):
        #edges = [(s1, neighbor) for neighbor in neighbors]
        if len(neighbors) == 0:
            pass
        else:
            for neighbor in neighbors:
                self.G.edges[(s1, neighbor)]["age"] += 1
        
    def error_update(self, s1, distance):        
        self.G.nodes[s1]["error"] -= distance[0]
        #self.G.nodes[s1]["SE"] += distance[0] 

    def position_update(self, s1, neighbors, X):
        ind_s1 = self.node_to_matrix_indice(s1)
        position_s1 = self.position_matrix[ind_s1,:]
        self.position_matrix[ind_s1,:] = self.learning_cible* (X - position_s1)

        for neighbor in neighbors:
            indice_neighbor =  self.node_to_matrix_indice(neighbor)
            position_neighbor = self.position_matrix[indice_neighbor,:]
            self.position_matrix[indice_neighbor, :] = self.learning_neighbors * (X - position_neighbor)

    def update_link(self, s1, s2):
        if (s1, s2) in self.G.edges:
            self.G.edges[(s1, s2)]["age"] = 0
        else:
            self.G.add_edge(s1, s2, age = 0)

        for edge in self.G.edges:
            if self.G.edges[edge]["age"] >= self.threshold_age_edge:
                self.G.remove_edge(edge[0], edge[1])
        
        neurons_to_remove = [neuron for (neuron, value) in list(nx.degree(self.G)) if value == 0]
        for neuron in neurons_to_remove:
            indice_neurone = self.node_to_matrix_indice(neuron)
            self.G.remove_node(neuron)
            self.position_matrix = np.delete(self.position_matrix, indice_neurone, 0)

    def add_neuron(self):

        # D'abord on récupère les noeuds et les erreurs
        self.node_max += 1
        list_error = [(node,error) for (node,error) in nx.get_node_attributes(self.G,"error").items()]
        error = rankdata(np.array([error for (node, error) in list_error]),method="ordinal")
        nodes = np.array([node for (node, error) in list_error])

        # Ensuite on identifie le noeud avec le plus d'erreur
        bad_node1 = nodes[error == 1][0]
        #print(bad_node1)
        
        # On identifie parmis ses voisins le deuxième plus mauvais neurone
        bad_node_neighbors = [(node, self.G.nodes[node]["error"]) for node in nx.neighbors(self.G,bad_node1)]
        bad_node_neighbors = sorted(bad_node_neighbors, key = lambda a : a[1])
        bad_node2 = bad_node_neighbors[0][0]

        # On récupère leurs indexes
        index_node1, index_node2 = self.node_to_matrix_indice(bad_node1), self.node_to_matrix_indice(bad_node2)

        # On détermine la position du nouveau neurone
        new_node = self.node_max
        self.G.add_node(new_node, error=0)
        position1, position2 = self.position_matrix[index_node1,:], self.position_matrix[index_node2,:]
        position_new_node = np.reshape((0.5*(position1+position2)), (1,-1))
        self.position_matrix = np.vstack((self.position_matrix, position_new_node))
                
        # On ajoute et retire les liens qu'il faut ajouter et enlevé
        self.G.remove_edge(bad_node1, bad_node2)
        self.G.add_edge(bad_node1, new_node, age = 0)
        self.G.add_edge(bad_node2, new_node, age = 0)

    def error_damping(self):
        for node in self.G.nodes:
            self.G.nodes[node]["error"] *= self.error_damping_factor

    def node_to_matrix_indice(self, node):
        indice = np.array(range(self.position_matrix.shape[0]))[np.array(list(self.G.nodes)) == node]

        return indice
    
    def error_stopping(self, bic):

        if self.best_bic < bic:
            self.error_memory += 1
        else:
            self.error_memory = 0
            self.best_bic = bic

        return self.error_memory > self.tolerance
    
    def BIC(self,distances, rangs):
        
        # Nombre de paramètres
        k = self.position_matrix.shape[0]
        
        # Nombre de données (pour BIC -> k*np.log(n))
        n = self.position_matrix.shape[1]

        # Vraisemblance
        L = self.likelihood(distances, rangs)

        self.information_score = 2*k - 2*np.log(L)
        return self.information_score
    
    def likelihood(self, distances, rangs):
        """
            A. On calcul la distance d'un point A à tous les noeuds du graphe D_A
            B. On divise la distance par la somme des distances d'un point à tous les noeuds DD = D_A/sum(D_A)
            C. L = 1-DD


            A. La probabilité qu'un noeud soit le bon noeud c'est 1-D, où D est la distance normalisée (1 étant le max et 0 le min)
            B. La vraisemblance c'est la vraisemblance d'une loi normale dont l'écart-type est celui des distances associées au noeud
            C. La likelihood globale c'est P(d_i|theta_j)P(theta_j) où
                    P(d_i|theta_j) : c'est la vraisemblance de la distance sachant la loi normale
                    P(theta_j) : C'est à quel point ce noeud est pertinent

            Une mixture est définie par
                P(x) = sum {theta_i * p_i(x)}

                Soit la somme de la vraisemblance des différentes distributions, fois le poid de chacune
        """

        ### A - Calcul de la probabilité de chaque noeuds pour chaque points
        inv_distances = np.exp(-(distances+0.001))
        p_noeuds = inv_distances/inv_distances.sum(0,keepdims=True)

        ### B - Calcul de l'écart type pour la distribution normale encodant les données
        std_noeuds = np.std(distances,axis=0,where=rangs==1)

        ### C - Calcul de la vraisemblance pour chaque x
        vraisemblance = []
        for e,std in enumerate(std_noeuds):
            vraisemblance.append(np.prod(norm(0,std).pdf(distances[e])*p_noeuds[e]))

        ### D - Calcul du produit des vraisemblances
        vraisemblance = np.prod(vraisemblance)
        print(vraisemblance)
        return vraisemblance
    

#GNG = GrowingNeuralGas(lambda_constante=100,
#learning_cible = 0.2, learning_neighbors=0.006, error_damping_factor=0.995, threshold_age_edge=50 )

from tqdm import tqdm 
def dynamic_graph(G_in,  time_series, position_matrix):

    transition = []
    for time_serie in tqdm(time_series):
        cluster_serie = get_cluster_ts(time_serie, position_matrix, [node for node in G_in.nodes])
        #print(cluster_serie)
        transition += ["-".join((str(x[0]),str(x[1]))) for x in sliding_window_view(cluster_serie, 2)]
    
    #print(transition[0])
    un, count = np.unique(transition, return_counts=True)
    #print(un)
    G_out = nx.DiGraph()
    G_out.add_weighted_edges_from([(a.split("-")[0], a.split("-")[1],b) for a,b in zip(un,count)])

    return G_out


def get_cluster_ts(ts_in, position_cluster, nodes):

    serie_out = []
    for step in ts_in:
        step = np.reshape(step, (1,-1)).astype(np.float64)
        #print(step.shape, position_cluster.shape)
        best_cluster = np.array(nodes)[rankdata(cdist(step,position_cluster), method="ordinal") == 1][0]
        
        serie_out.append(best_cluster)
    
    return serie_out
        

def get_list_timeserie(df_in):
    lorenz_list = []
    for r in tqdm(range(df_in.run.max())):
        lorenz_list.append(df_in[df_in.run == r].loc[:,["x","y","z"]].values)
    return lorenz_list




class GrowingNeuralGas1():
    
    def __init__(self,
                 lambda_constante,
                 learning_cible,
                 learning_neighbors,
                 error_damping_factor,
                 threshold_age_edge,
                 threshold_error = 100_000,
                 nb_iteration = 500_000,
                 fraction_node = 0.01,
                 nb_node = 500,
                 tolerance=50,
                 growing=True,
                 ) -> None:
        
        

        self.node_max = 1

        # Hyper parameter
        self.growing = growing
        
        self.nb_nodes_init = 50
        self.lambda_constante = lambda_constante
        self.learning_cible = learning_cible
        self.learning_neighbors = learning_neighbors
        self.error_damping_factor = error_damping_factor
        self.threshold_age_edge = threshold_age_edge
        self.initialize_graph()
        # Early Stopping parameters
        self.iteration_criterion = nb_iteration
        self.fraction_node = fraction_node
        self.nb_node = nb_node
        self.error_thresh = threshold_error
        self.error_memory = 0
        self.prev_error = 0
        self.tolerance = tolerance

        self.iterate = 1
        
    def initialize_graph(self):
        self.G = nx.Graph()
        if self.growing:
            self.G.add_nodes_from([0,1])
        else:
            self.G.add_nodes_from(list(range(self.nb_nodes_init)))
        nx.set_node_attributes(self.G, {node : 0 for node in self.G.nodes}, name="error")
        nx.set_node_attributes(self.G, {node : 0 for node in self.G.nodes}, name="SE")
        nx.set_node_attributes(self.G, {node : 0 for node in self.G.nodes}, name="n_memb")
    

    def init_fit(self,X_train):
        if self.growing:
            self.position_matrix = np.random.uniform(-1,1,size=(2,X_train.shape[1]))
        else:
            self.position_matrix = np.random.uniform(-1,1,size=(self.nb_nodes_init,X_train.shape[1]))

    def fit(self, X_train, more_iteration = 100):
        
        if self.iterate == 0:
            self.iterate = 1
            self.iteration_criterion = more_iteration
            pbar = tqdm(total=more_iteration)
            self.error_memory = 0
        else : 
            self.init_fit(X_train)
            pbar = tqdm(total=self.iteration_criterion)
        
        while self.iterate != 0:

            # 0 - Get the data
            ind_X = random.choice([i for i in range(X_train.shape[0])])
            X = np.reshape(X_train[ind_X,:],(1,-1))

            #print(X.shape, self.position_matrix.shape)
            # 1 - On calcul les distances et on classe les noeuds du plus proche au moins proche
            distances = np.reshape(cdist(X,self.position_matrix),(-1))
            rank_distance = rankdata(distances,method="ordinal")
            #print(rank_distance.shape)
            #print(distances, rank_distance)
            # 2 - On identifie les deux noeuds qui sont connectés
            s1 = np.array(self.G.nodes)[rank_distance==1][0]
            s2 = np.array(self.G.nodes)[rank_distance==2][0]
            #print(s1)
            # 3 - On update l'age des liens
            self.age_update(s1, [neighbor for neighbor in self.G.neighbors(s1)])

            # 4 - On update le taux d'erreur du noeud
            self.error_update(s1, distances[rank_distance==1])

            # 5 - On déplace le noeud et ses voisins
            self.position_update(s1,[neighbor for neighbor in self.G.neighbors(s1)], X)

            # 6 - On met à jours les liens et retire les noeuds non connectés
            self.update_link(s1,s2)

            # 7 - On ajoute un noeud si il le faut
            if self.growing:
                if self.iterate%self.lambda_constante == 0:
                    self.add_neuron()

            # 8 - Error update
            self.error_damping()

            # 9 - Early_stopping criterion
            if self.iterate == self.iteration_criterion:
                break
            elif len(list(self.G.nodes)) > 0.01*X_train.shape[0]:
                break 
            elif len(list(self.G.nodes)) > self.nb_node:
                break
            elif self.error_stopping(distances[0]):
                break
            else:
                self.iterate += 1
            pbar.update(1)
            

    def age_update(self,s1, neighbors):
        #edges = [(s1, neighbor) for neighbor in neighbors]
        if len(neighbors) == 0:
            pass
        else:
            for neighbor in neighbors:
                self.G.edges[(s1, neighbor)]["age"] += 1
        
    def error_update(self, s1, distance):        
        self.G.nodes[s1]["error"] -= distance[0]
        self.G.nodes[s1]["SE"] += distance[0] 

    def position_update(self, s1, neighbors, X):
        ind_s1 = self.node_to_matrix_indice(s1)
        position_s1 = self.position_matrix[ind_s1,:]
        self.position_matrix[ind_s1,:] = self.learning_cible* (X - position_s1)

        for neighbor in neighbors:
            indice_neighbor =  self.node_to_matrix_indice(neighbor)
            position_neighbor = self.position_matrix[indice_neighbor,:]
            self.position_matrix[indice_neighbor, :] = self.learning_neighbors * (X - position_neighbor)

    def update_link(self, s1, s2):
        if (s1, s2) in self.G.edges:
            self.G.edges[(s1, s2)]["age"] = 0
        else:
            self.G.add_edge(s1, s2, age = 0)


        for edge in self.G.edges:
            if self.G.edges[edge]["age"] >= self.threshold_age_edge:
                self.G.remove_edge(edge[0], edge[1])
        
        if self.growing:
            neurons_to_remove = [neuron for (neuron, value) in list(nx.degree(self.G)) if value == 0]
            for neuron in neurons_to_remove:
                indice_neurone = self.node_to_matrix_indice(neuron)
                self.G.remove_node(neuron)
                self.position_matrix = np.delete(self.position_matrix, indice_neurone, 0)

    def add_neuron(self):

        # D'abord on récupère les noeuds et les erreurs
        self.node_max += 1
        list_error = [(node,error) for (node,error) in nx.get_node_attributes(self.G,"error").items()]
        error = rankdata(np.array([error for (node, error) in list_error]),method="ordinal")
        nodes = np.array([node for (node, error) in list_error])

        # Ensuite on identifie le noeud avec le plus d'erreur
        bad_node1 = nodes[error == 1][0]
        #print(bad_node1)
        
        # On identifie parmis ses voisins le deuxième plus mauvais neurone
        bad_node_neighbors = [(node, self.G.nodes[node]["error"]) for node in nx.neighbors(self.G,bad_node1)]
        bad_node_neighbors = sorted(bad_node_neighbors, key = lambda a : a[1])
        bad_node2 = bad_node_neighbors[0][0]

        # On récupère leurs indexes
        index_node1, index_node2 = self.node_to_matrix_indice(bad_node1), self.node_to_matrix_indice(bad_node2)

        # On détermine la position du nouveau neurone
        new_node = self.node_max
        self.G.add_node(new_node, error=0)
        position1, position2 = self.position_matrix[index_node1,:], self.position_matrix[index_node2,:]
        position_new_node = np.reshape((0.5*(position1+position2)), (1,-1))
        self.position_matrix = np.vstack((self.position_matrix, position_new_node))
                
        # On ajoute et retire les liens qu'il faut ajouter et enlevé
        self.G.remove_edge(bad_node1, bad_node2)
        self.G.add_edge(bad_node1, new_node, age = 0)
        self.G.add_edge(bad_node2, new_node, age = 0)


    def error_damping(self):
        for node in self.G.nodes:
            self.G.nodes[node]["error"] *= self.error_damping_factor

    def node_to_matrix_indice(self, node):
        indice = np.array(range(self.position_matrix.shape[0]))[np.array(list(self.G.nodes)) == node]

        return indice
    def error_stopping(self, distance):
        if self.prev_error > distance:
            self.error_memory += 1
        else:
            self.error_memory = 0

        return self.error_memory > self.tolerance
    

#from growing_neural_gas import GrowingNeuralGas

"""
GNG_true = GrowingNeuralGas1(lambda_constante=100,learning_cible = 0.2, learning_neighbors=0.006, error_damping_factor=0.995, threshold_age_edge=50,nb_node=100,nb_iteration=1_000,growing=False)
GNG_true.fit(np.nan_to_num((new_df.droplevel(0).groupby(level=0).head(1)+df_parquet.groupby(level=1).cumsum(1))))
"""


class NeuralGas():

    def __init__(self,
                 trial=None,
                 dict_param=None,
                 n_nodes=50,
                 n_iteration = 2_500,
                 lambda_init = 15,
                 lambda_final=0.01,
                 eta_init=0.3,
                eta_final = 0.05,
                age_threshold_init=20,
                age_threshold_final=200,
                update=True

                 ) -> None:
        
        self.trial = trial
        self.dict_param = dict_param

        if dict_param == None:
            self.n_nodes = self.trial.suggest_int("n_nodes",50,500)

            self.lambda_init = self.trial.suggest_int("lambda_init",1,50)
            self.lambda_final = self.trial.suggest_loguniform("lambda_final",1e-6,1)
            

            self.eta_init = self.trial.suggest_loguniform("eta_init",1e-6,1)
            self.eta_final = self.trial.suggest_loguniform("eta_final",1e-7,self.eta_init)
            

            self.age_threshold_init = self.trial.suggest_int("age_threshold_init",5,50)
            self.age_threshold_final = self.trial.suggest_int("age_threshold_final",self.age_threshold_init,500)
        else:
            self.n_nodes = dict_param["n_nodes"]

            self.lambda_init = dict_param["lambda_init"]
            self.lambda_final = dict_param["lambda_final"]

            self.eta_init = dict_param["eta_init"]
            self.eta_final = dict_param["eta_final"]

            self.age_threshold_init = dict_param["age_threshold_init"]
            self.age_threshold_final = dict_param["age_threshold_final"]


        self.lambada = self.lambda_init
        self.eta = self.eta_init 
        self.age_threshold = self.age_threshold_init
        self.init_graph()

        self.n_iteration = n_iteration
        self.update=update
        self.aic=[]

    def init_graph(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(list(range(self.n_nodes)))

    def get_pb_articles(self, X_train):
        # 1 - get the frequency of an index
        # 2 - Compute the inverse of it 
        # 3 - Assign a pb of it to be drawn 

        return None 

    def fit(self,X_train):
        
        
        random_index = random.sample(list(range(X_train.shape[0])), k=self.n_nodes)
        self.position_matrix = np.array([X_train[ind,:] for ind in random_index])
        #self.position_matrix = np.random.uniform(-5,5,size=(self.n_nodes,X_train.shape[1]))

        #for i in tqdm(range(self.n_iteration)):
        for i in range(self.n_iteration):
            # 0 - Get the data
            ind_X = random.choice([j for j in range(X_train.shape[0])])
            
            X = np.reshape(X_train[ind_X,:],(1,-1))

            #print(X.shape, self.position_matrix.shape)
            # 1 - On calcul les distances et on classe les noeuds du plus proche au moins proche
            distances = np.reshape(cdist(X,self.position_matrix),(-1))
            rank_distance = np.argsort(distances)+1

            # 2 - Update position
            self.update_position(rank_distance, X)

            # 3 - Update links
            s1 = np.array(self.G.nodes)[rank_distance==1][0]
            s2 = np.array(self.G.nodes)[rank_distance==2][0]
            self.update_link(s1,s2)

            # 4 - Links again
            self.age_update(s1, [neighbor for neighbor in self.G.neighbors(s1)])

            if self.update:
                self.parameter_update(i)

            if i%100 == 0:
                ind_X = random.choices([i for i in range(X_train.shape[0])],k=1_000)
                X = X_train[ind_X,:]

                self.AIC(X)
                
                if self.dict_param == None:
                    self.trial.report(self.aic[-1], i)

                    if self.trial.should_prune():
                        raise optuna.TrialPruned()
            
            
    def update_position(self, rank_distances, v):
        variation = (v - self.position_matrix)
        variation_magnitude = self.eta * np.exp(-rank_distances/self.lambada).reshape((-1,1))
        #print(variation.shape, variation_magnitude.shape, self.position_matrix.shape)

        self.position_matrix = np.nan_to_num(self.position_matrix + variation * variation_magnitude)

    def update_link(self, s1, s2):
        if (s1, s2) in self.G.edges:
            self.G.edges[(s1, s2)]["age"] = 0
        else:
            self.G.add_edge(s1, s2, age = 0)


        for edge in self.G.edges:
            if self.G.edges[edge]["age"] >= self.age_threshold:
                self.G.remove_edge(edge[0], edge[1])

    def age_update(self,s1, neighbors):
        #edges = [(s1, neighbor) for neighbor in neighbors]
        if len(neighbors) == 0:
            pass
        else:
            for neighbor in neighbors:
                self.G.edges[(s1, neighbor)]["age"] += 1
                if self.G.edges[(s1, neighbor)]["age"] > self.age_threshold:
                    self.G.remove_edge(s1, neighbor)

    def parameter_update(self,step):

        fct_update = lambda init, final, step, tmax : init*np.power(final/init,(1+step)/tmax)
        if step == self.n_iteration:
            self.lambada = self.lambda_final
            self.eta = self.eta_final
            self.age_threshold = self.age_threshold_final
        else:
            self.lambada = fct_update(self.lambda_init, self.lambda_final, step, self.n_iteration)
            self.eta = fct_update(self.eta_init, self.eta_final, step, self.n_iteration)
            self.lambada = fct_update(self.age_threshold_init, self.age_threshold_final, step, self.n_iteration)

    def save_NG(self,path):
        with open(path+"/graph.pkl","wb") as f:
            pickle.dump(self.G,f)
        
        np.save(path+"/position_matrix.npy",self.position_matrix)

    def load_NG(self, path):
        with open(path+"/graph.pkl","rb") as f:
            self.G = pickle.load(f)
        
        self.position_matrix = np.load(path+"/position_matrix.npy")

    def AIC(self,X):
        distances = np.power(cdist(X,self.position_matrix),2)
        rangs = rankdata(distances,axis=1, method="ordinal")
        RSS = distances[rangs==1].sum()

        score = 2*self.position_matrix.shape[0]+X.shape[0]*np.log(RSS+X.shape[0])
        self.aic.append(score)

        
#NG = NeuralGas()
#NG.fit(np.nan_to_num((new_df.droplevel(0).groupby(level=0).head(1)+df_parquet.groupby(level=1).cumsum(1))))


def entraineur_NG(X):
    """
        On utilise Optuna pour trouver les meilleurs paramètre d'entraînement
    """
    pass


class GaussianGrowingNeuralGas():
    """

    """
    def __init__(self,
                 trial,
                 dict_params=None,
                 threshold_error = 15,
                 nb_iteration = 10_000,
                 nb_node = 500,
                 tolerance=50,
                 dir="/GNG_gaussien",
                 p_matrice="position_matrix.npy",
                 p_graphe="graphe.adjlist",
                    opti=True,
                    display_pbar=False            
                 ) -> None:
        
        
        self.trial = trial
        self.dict_params = dict_params
        self.path_dir=dir
        self.path_G=dir+"/"+p_graphe 
        self.path_matrix = dir+"/"+p_matrice

        
        

        # Hyper parameter
        if dict_params == None:
            self.nb_nodes_init = self.trial.suggest_int("nb_nodes_init",2,50)
            self.lambda_constante = self.trial.suggest_int("lambda_constante",5,50)
            self.learning_cible = self.trial.suggest_loguniform("learning_cible",1e-6,100)
            self.learning_neighbors = self.trial.suggest_loguniform("learning_neighbors",1e-6,100)
            self.error_damping_factor = self.trial.suggest_loguniform("damping_error",1e-6,1)
            self.threshold_age_edge = self.trial.suggest_int("thresholg_age_link",10,500)
        else:
            self.nb_nodes_init = self.dict_params["nb_nodes_init"]
            self.lambda_constante = self.dict_params["lambda_constante"]
            self.learning_cible = self.dict_params["learning_cible"]
            self.learning_neighbors = self.dict_params["learning_neighbors"]
            self.error_damping_factor = self.dict_params["damping_error"]
            self.threshold_age_edge = self.dict_params["thresholg_age_link"]

        self.initialize_graph()
        # Early Stopping parameters
        self.iteration_criterion = nb_iteration

        self.nb_node = nb_node
        self.error_memory = 0
        self.prev_error = 0
        self.tolerance = tolerance
        self.aic = []

        self.opti=opti
        self.node_max = self.nb_nodes_init
        self.display_pbar = display_pbar
        self.iterate = 1
        
    def initialize_graph(self):
        """
            - Initialise le graphe
                1. Créer le graphe avec autant de noeuds que demandés
                2. Ajoute à chaque noeuds un paramètre d'erreur
        """

        self.G = nx.Graph()
        self.G.add_nodes_from(list(range(self.nb_nodes_init)))

        nx.set_node_attributes(self.G, {node : 0 for node in self.G.nodes}, name="error")
        nx.set_node_attributes(self.G, {node : 0 for node in self.G.nodes}, name="sd")
    

    def init_fit(self,X_train):
        """
            - Initialise la matrice des noeuds
        """
        self.position_matrix = np.random.uniform(-10,10,size=(self.nb_nodes_init,X_train.shape[1]))

    def fit(self, X_train, more_iteration = 100):
        
        # Pour reprendre l'entraînement avec un autre jeu de données je suppose
        if self.iterate == 0:
            self.iterate = 1
            self.iteration_criterion = more_iteration
            if self.display_pbar:
                pbar = tqdm(total=more_iteration)
            self.error_memory = 0
        else : 
            self.init_fit(X_train)
            if self.display_pbar:
                pbar = tqdm(total=self.iteration_criterion)
        

        for it in range(self.iteration_criterion):

            # 0 - Get the data
            ind_X = random.choice([i for i in range(X_train.shape[0])])
            X = np.reshape(X_train[ind_X,:],(1,-1))


            # 1 - On calcul les distances et on classe les noeuds du plus proche au moins proche
            distances = np.reshape(cdist(X,self.position_matrix),(-1))
            rank_distance = rankdata(distances,method="ordinal")

            # 2 - On identifie les deux noeuds qui sont connectés
            try:
                s1 = np.array(self.G.nodes)[rank_distance==1][0]
                s2 = np.array(self.G.nodes)[rank_distance==2][0]
            except Exception:
                print(self.position_matrix)
                print(self.G.nodes)
                print(rank_distance)
                print(distances)
                self.iterate == 0
                break 

            # 3 - On update l'age des liens
            self.age_update(s1, [neighbor for neighbor in self.G.neighbors(s1)])

            # 4 - On update le taux d'erreur du noeud
            self.error_update(s1, distances[rank_distance==1],[neighbor for neighbor in self.G.neighbors(s1)])

            # 5 - On déplace le noeud et ses voisins
            self.position_update(s1,[neighbor for neighbor in self.G.neighbors(s1)], X)

            # 6 - On met à jours les liens et retire les noeuds non connectés
            self.update_link(s1,s2)

            # 7 - On ajoute un noeud si il le faut
            
            if self.iterate%self.lambda_constante == 0:
                self.add_neuron()

            # 8 - Error update
            self.error_damping()


            # 9 - Early_stopping criterion
            """
                1. Sélectionne 100 points
                2. Calcul la somme des erreurs
                3. Calcul l'AIC
                4. Stop si l'AIC ne s'améliore pas
            """
            ### 9.1 : Sélection des points
            ind_X = random.choices([i for i in range(X_train.shape[0])],k=100)
            X = X_train[ind_X,:]

            ### 9.2 / 9.3 : Calcul de l'AIC
            self.AIC(X)
            self.trial.report(self.aic[-1], self.iterate)

            if self.trial.should_prune():
                raise optuna.TrialPruned()
            
            ### 9.4 : Stopping criterion
            if self.iterate == self.iteration_criterion:
                self.iterate==0
                break
            elif self.error_stopping():
                self.iterate==0
                break
            else:
                self.iterate += 1

            # 10 - Sauve le modèle s'il est le meilleur actuellement
            if not self.opti:
                self.save_model()
            if self.display_pbar:
                pbar.update(1)
            

    def age_update(self,s1, neighbors):
        """
            Augmente l'age du liens entre le noeud s1 et ses voisins, s'il en a
        """
        #edges = [(s1, neighbor) for neighbor in neighbors]
        if len(neighbors) == 0:
            pass
        else:
            for neighbor in neighbors:
                self.G.edges[(s1, neighbor)]["age"] += 1
        
    def error_update(self, s1, distance, neighbors):        
        """
            Met à jours l'erreur, on fait -= parce que le rankdata rank dans l'autre sens ! Donc on veut l'unité avec le score le plus négatif

            Pour calculer l'erreur
                1. On calcul met à jours la sd du noeud
        """
        # 1 : Récupère le vecteur
        ind_s1 = self.node_to_matrix_indice(s1)
        position_s1 = self.position_matrix[ind_s1,:]

        # 2 : Calcul SD
        sd = 0
        for neighbor in neighbors:
            indice_neighbor =  self.node_to_matrix_indice(neighbor)
            position_neighbor = self.position_matrix[indice_neighbor,:]
            sd += np.linalg.norm(position_s1-position_neighbor)/len(neighbors)

        # Erreur = logcdf, ça va de -0.7 si la distance est de 0; à 0 si la distance est de plus l'infini
        self.G.nodes[s1]["error"] -= norm(0,sd+0.001).logcdf(distance[0])
 

    def position_update(self, s1, neighbors, X):
        """
            1. Récupère le vecteur du meilleur noeud
            2. M
        """
        # 1 : Récupère le vecteur
        ind_s1 = self.node_to_matrix_indice(s1)
        position_s1 = self.position_matrix[ind_s1,:]

        # 2 : Met à jours la position 
        self.position_matrix[ind_s1,:] = self.learning_cible* (X - position_s1)

        # 3 : Fait la même chose avec les voisins
        for neighbor in neighbors:
            indice_neighbor =  self.node_to_matrix_indice(neighbor)
            position_neighbor = self.position_matrix[indice_neighbor,:]
            self.position_matrix[indice_neighbor, :] = self.learning_neighbors * (X - position_neighbor)

    def update_link(self, s1, s2):
        if (s1, s2) in self.G.edges:
            self.G.edges[(s1, s2)]["age"] = 0
        else:
            self.G.add_edge(s1, s2, age = 0)


        for edge in self.G.edges:
            if self.G.edges[edge]["age"] >= self.threshold_age_edge:
                self.G.remove_edge(edge[0], edge[1])
        

        neurons_to_remove = [neuron for (neuron, value) in list(nx.degree(self.G)) if value == 0]
        for neuron in neurons_to_remove:
            indice_neurone = self.node_to_matrix_indice(neuron)
            self.G.remove_node(neuron)
            self.position_matrix = np.delete(self.position_matrix, indice_neurone, 0)

    def add_neuron(self):

        # D'abord on récupère les noeuds et les erreurs
        self.node_max += 1
        list_error = [(node,error) for (node,error) in nx.get_node_attributes(self.G,"error").items()]
        error = rankdata(np.array([error for (node, error) in list_error]),method="ordinal")
        nodes = np.array([node for (node, error) in list_error])
        
        # Ensuite on identifie le noeud avec le plus d'erreur
        bad_node1 = nodes[error == 1][0]
        #print(bad_node1)
        
        # On identifie parmis ses voisins le deuxième plus mauvais neurone
        bad_node_neighbors = [(node, self.G.nodes[node]["error"]) for node in nx.neighbors(self.G,bad_node1)]
        bad_node_neighbors = sorted(bad_node_neighbors, key = lambda a : a[1])
        bad_node2 = bad_node_neighbors[0][0]

        # On récupère leurs indexes
        index_node1, index_node2 = self.node_to_matrix_indice(bad_node1), self.node_to_matrix_indice(bad_node2)

        # On détermine la position du nouveau neurone
        new_node = self.node_max
        self.G.add_node(new_node, error=0, sd=0)
        position1, position2 = self.position_matrix[index_node1,:], self.position_matrix[index_node2,:]
        position_new_node = np.reshape((0.5*(position1+position2)), (1,-1))
        self.position_matrix = np.vstack((self.position_matrix, position_new_node))
                
        # On ajoute et retire les liens qu'il faut ajouter et enlevé
        self.G.remove_edge(bad_node1, bad_node2)
        self.G.add_edge(bad_node1, new_node, age = 0)
        self.G.add_edge(bad_node2, new_node, age = 0)


    def error_damping(self):
        for node in self.G.nodes:
            self.G.nodes[node]["error"] *= self.error_damping_factor

    def node_to_matrix_indice(self, node):
        indice = np.array(range(self.position_matrix.shape[0]))[np.array(list(self.G.nodes)) == node]

        return indice
    
    def error_stopping(self):
        if len(self.aic) <2 :
            self.error_memory += 1
        elif self.aic[-2] < self.aic[-1]:
            self.error_memory+=1
        else:
            self.error_memory = 0

        return self.error_memory > self.tolerance
    
    def AIC(self,X):
        distances = np.power(cdist(X,self.position_matrix),2)
        rangs = rankdata(distances,axis=1, method="ordinal")
        RSS = distances[rangs==1].sum()

        score = 2*self.position_matrix.shape[0]+X.shape[0]*np.log(RSS+X.shape[0])
        self.aic.append(score)

    def save_model(self):
        if self.aic[-1] == min(self.aic):
            if not os.path.exists(self.path_dir):
                os.makedirs(self.path_dir)
            
            nx.write_adjlist(self.G,self.path_G)
            np.save(self.position_matrix,self.path_matrix)
        