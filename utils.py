from lightning.pytorch.callbacks import TQDMProgressBar
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
import json

# for tqdm on screen so it can use ascii
class MyProgressBar(TQDMProgressBar):

    def init_sanity_tqdm(self) -> tqdm:
        bar = tqdm(
            desc=self.sanity_check_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout, ascii=True,
        )
        return bar

    def init_train_tqdm(self) -> tqdm:
        bar = tqdm(
            desc=self.train_description,
            #initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0, ascii=True,
        )
        return bar

    def init_predict_tqdm(self) -> tqdm:
        bar = tqdm(
            desc=self.predict_description,
            #initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0, ascii=True,
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        bar = tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout, ascii=True,
        )
        return bar

    def init_test_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout, ascii=True,
        )
        return bar


class config_ffnn:
    def __init__(self, id, num_layers, num_neurons, activation_funct, learning_rate, batch_size, neg_slope_leakyrelu=None, alpha_loss=None):
        self.id = id
        self.costs = []
        self.epochs = []
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.activation_funct = activation_funct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.neg_slope_leakyrelu = neg_slope_leakyrelu
        self.alpha_loss = alpha_loss
    def add_cost(self, cost):
        self.costs.append(cost)
    def add_epoch(self, epoch):
        self.epochs.append(epoch)
    def print(self):
        print("id: ",self.id,"\nnum_layers: ", self.num_layers)
        print("num_neurons: ", self.num_neurons[:self.num_layers])
        print("activation_functions: ", self.activation_funct[:self.num_layers])
        print("lr: ", self.learning_rate)
        print("bs: ", self.batch_size)
        print("Costs: ", self.get_best_cost())
        print("Epochs: ",self.get_best_epoch())
    def config_dict(self):
        config_dict = {
            "ID": self.id,
            "batch_size": self.batch_size,
            "learning_rate_init": self.learning_rate,
            "n_layer": self.num_layers,
            "neg_slope_leakyrelu": self.neg_slope_leakyrelu}
        for i in range(1, self.num_layers+1):
            config_dict["n_neurons_"+str(i)] = self.num_neurons[i-1]
            config_dict["activation_"+str(i)] = self.activation_funct[i-1]
        return config_dict
    def get_best_cost(self):
        return min(self.costs)
    def get_best_epoch(self):
        return self.epochs[self.costs.index(min(self.costs))]

class config_transformer:
    def __init__(self, id, num_layers, embed_dim, hidden_dim, learning_rate, batch_size, num_heads, dropout):
        self.id = id
        self.costs = []
        self.epochs = []
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.dropout = dropout
    def add_cost(self, cost):
        self.costs.append(cost)
    def add_epoch(self, epoch):
        self.epochs.append(epoch)
    def print(self):
        print("id: ",self.id,"\nnum_layers: ", self.num_layers," - embed_dim: ", self.embed_dim," - hidden_dim: ", self.hidden_dim," - num_heads: ",self.num_heads, " - lr: ", self.learning_rate," - bs: ", self.batch_size," - dropout: ", self.dropout)
        print("Costs: ", self.costs)
        print("Epochs: ",self.epochs)
    def config_dict(self):
        config_dict = {
            "ID": self.id,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,}
        return config_dict
    def get_best_cost(self):
        return min(self.costs)
    def get_best_epoch(self):
        return self.epochs[self.costs.index(min(self.costs))]

class config_xgboost:
    def __init__(self, id, eta, n_estimators, gamma, max_depth, min_child_weight, max_delta_step, subsample, colsample_bytree, colsample_bylevel, colsample_bynode,):
        self.id = id
        self.costs = []
        self.epochs = []
        self.eta = eta
        self.n_estimators = n_estimators
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode


    def add_cost(self, cost):
        self.costs.append(cost)
    def print(self):
        print("id: ",self.id,"\neta: ", self.eta,"\nn_estimators: ", self.n_estimators," - gamma: ", self.gamma," - max_depth: ", self.max_depth," - min_child_weight: ",self.min_child_weight, " - max_delta_step: ", self.max_delta_step," - subsample: ", self.subsample," - colsample_bytree: ", self.colsample_bytree," - colsample_bylevel: ", self.colsample_bylevel," - colsample_bynode: ", self.colsample_bynode)
        print("Costs: ", self.costs)
    def config_dict(self):
        config_dict = {
            "eta": self.eta,
            "n_estimators": self.n_estimators,
            "gamma": self.gamma,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "max_delta_step": self.max_delta_step,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "colsample_bynode": self.colsample_bynode,}
        return config_dict
    def get_best_cost(self):
        return min(self.costs)


# this function is used to extract best smac_values
def BestSMAC_FFNN(file_name):
    # read jsonfile
    filepath= file_name+"/0/runhistory.json"

    jsonfile = open(filepath, 'r')
    jsondata = jsonfile.read()
    jsonobj = json.loads(jsondata)

    configs_names=list(jsonobj["configs"]["1"].keys())

    # search for all strings starting with "activation_fun"
    max_num_layers=len([s for s in configs_names if "activation_" in s])

    runs_history = []
    for id in jsonobj["configs"]:
        activations=[]
        num_neurons=[]
        for j in range(1,max_num_layers+1):
            activations.append(str(jsonobj["configs"][id]["activation_"+str(j)]))
            num_neurons.append(int(jsonobj["configs"][id]["n_neurons_"+str(j)]))
        single_config = config_ffnn(id,jsonobj["configs"][id]["n_layer"],num_neurons,activations,jsonobj["configs"][id]["learning_rate_init"],jsonobj["configs"][id]["batch_size"],neg_slope_leakyrelu=jsonobj["configs"][id]["neg_slope_leakyrelu"] if "neg_slope_leakyrelu" in jsonobj["configs"][id] else None,alpha_loss=jsonobj["configs"][id]["alpha_loss"] if "alpha_loss" in jsonobj["configs"][id] else None)
        runs_history.append(single_config)

    for i in jsonobj["data"]:
        runs_history[i[0]-1].add_cost(i[4])
        runs_history[i[0]-1].add_epoch(np.ceil(i[3]).astype(int))    

    all_best_costs=[]
    for i in range(len(runs_history)):
        all_best_costs.append(runs_history[i].get_best_cost())

    id_sorted=np.argsort(all_best_costs)
    best_config_id = id_sorted[0]
    best_config_dict = runs_history[best_config_id].config_dict()
    return best_config_dict

def BestSMAC_Transformer(file_name):
    # read jsonfile
    filepath= file_name+"/0/runhistory.json"

    jsonfile = open(filepath, 'r')
    jsondata = jsonfile.read()
    jsonobj = json.loads(jsondata)

    # search for all strings starting with "activation_fun"

    runs_history = []
    for id in jsonobj["configs"]:
        single_config = config_transformer(id,jsonobj["configs"][id]["num_layers"],jsonobj["configs"][id]["embed_dim"],jsonobj["configs"][id]["hidden_dim"],jsonobj["configs"][id]["learning_rate"],jsonobj["configs"][id]["batch_size"],jsonobj["configs"][id]["num_heads"],jsonobj["configs"][id]["dropout"])
        runs_history.append(single_config)

    for i in jsonobj["data"]:
        runs_history[i[0]-1].add_cost(i[4])
        runs_history[i[0]-1].add_epoch(np.ceil(i[3]).astype(int))    

    all_best_costs=[]
    for i in range(len(runs_history)):
        all_best_costs.append(runs_history[i].get_best_cost())

    id_sorted=np.argsort(all_best_costs)
    best_config_id = id_sorted[0]
    best_config_dict = runs_history[best_config_id].config_dict()
    return best_config_dict

def BestSMAC_XGBoost(file_name):
    # read jsonfile
    filepath= file_name+"/0/runhistory.json"

    jsonfile = open(filepath, 'r')
    jsondata = jsonfile.read()
    jsonobj = json.loads(jsondata)

    # search for all strings starting with "activation_fun"

    runs_history = []
    for id in jsonobj["configs"]:
        single_config = config_xgboost(id,jsonobj["configs"][id]["eta"],jsonobj["configs"][id]["n_estimators"],jsonobj["configs"][id]["gamma"],jsonobj["configs"][id]["max_depth"],jsonobj["configs"][id]["min_child_weight"],jsonobj["configs"][id]["max_delta_step"],jsonobj["configs"][id]["subsample"],jsonobj["configs"][id]["colsample_bytree"],jsonobj["configs"][id]["colsample_bylevel"],jsonobj["configs"][id]["colsample_bynode"])
        runs_history.append(single_config)

    for i in jsonobj["data"]:
        runs_history[i[0]-1].add_cost(i[4])

    all_best_costs=[]
    for i in range(len(runs_history)):
        all_best_costs.append(runs_history[i].get_best_cost())

    id_sorted=np.argsort(all_best_costs)
    best_config_id = id_sorted[0]
    best_config_dict = runs_history[best_config_id].config_dict()
    return best_config_dict

