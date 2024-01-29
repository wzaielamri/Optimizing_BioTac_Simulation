#!/usr/bin/env python

# for the mae + mse loss
# old one works good

# this is with forces no window +10-10: SMAC
Best_config = {
      "activation_1": "tanh",
      "activation_2": "relu",
      "activation_3": "elu",
      "activation_4": "leakyrelu",
      "activation_5": "elu",
      "activation_6": "relu",
      "activation_7": "leakyrelu",
      "batch_size": 512,
      "learning_rate_init": 0.0003,
      "n_layer": 7,
      "n_neurons_1": 820,
      "n_neurons_2": 740,
      "n_neurons_3": 190,
      "n_neurons_4": 740,
      "n_neurons_5": 1000,
      "n_neurons_6": 850,
      "n_neurons_7": 430,
      "neg_slope_leakyrelu": 0.1,
}

import numpy as np
import csv
import os
import sys
import logging
import utils
from os.path import basename
import time
import psutil

from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import wandb

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch.optim as optim
from torch.utils.data import Dataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--window_before', type=int, default=10, help='window before')
parser.add_argument('--window_after', type=int, default=10, help='window after')
parser.add_argument('--specific', type=bool, default=False, help='specific')
parser.add_argument('--next_pos', type=bool, default=False, help='next_pos')
parser.add_argument('--last_pos', type=bool, default=False, help='last_pos')
parser.add_argument("--file", type=str, default="./biotac_single_contact_response/2018-01-19-18-16-58_biotac_ff_stick_calibration.bag.csv", help="file path")
parser.add_argument("--prefix", type=str, default="", help="prefix")
parser.add_argument('--load_smac', type=bool, default=False, help='load_smac')
parser.add_argument('--add_five', type=bool, default=False, help='add_five')


args = parser.parse_args()

seed=args.seed
np.random.seed(seed)
pl.seed_everything(seed, workers=True)



exp_name="FFNN_SMACBest_Pytorch"
if args.specific:
    exp_name += "_No_Window_-"+str(args.window_before)+"+"+str(args.window_after)
else: 
    exp_name += "_Window_-"+str(args.window_before)+"+"+str(args.window_after)
if args.last_pos:
    exp_name += "_LastPos"
if args.next_pos:
    exp_name += "_NextPos"
exp_name += "_Inference"+str(args.prefix)

wandb_logger = WandbLogger(log_model="all",    
    project="BioTacPlugin",
    group="DDP",
    # run_name
    name= exp_name,

    # track hyperparameters and run metadata with wandb.config
    config={
        "note": "SMACBest Setting: TrainValTest dataset Split Pytorch EralyStop FFNN "+str(args.prefix),
        "optimizer": "Adam",
        "loss": "default_loss",
    })


if args.load_smac:
    smac_path="./smac3_output/SMACSearch_FFNN_pytorch"
    if args.specific:
        smac_path += "_No_Window_-"+str(args.window_before)+"+"+str(args.window_after)
    else: 
        smac_path += "_Window_-"+str(args.window_before)+"+"+str(args.window_after)
    if args.last_pos:
        smac_path += "_LastPos"
    if args.next_pos:
        smac_path += "_NextPos"
    smac_path += "_testtrain"+str(args.prefix)

    # extract best_config
    Best_config=utils.BestSMAC_FFNN(smac_path)

print(exp_name)
print(smac_path)
exit()


print("Best_config: ", Best_config)

# save best_config
wandb_logger.experiment.config.update(Best_config)


file_path = args.file
reader = csv.reader(open(file_path))
rows = [row for row in reader]


data_columns = [7, 9, ] + list(range(12, 31))

headers_out = np.array(rows[0])[data_columns].tolist()

rows = rows[1:]
data = np.array(rows).astype(float)


#closest_points = np.load('stick_closest_points.npy', allow_pickle=True)
# chnage the x,y,z with closest_points
#data[:,1:4] = closest_points


last_position = [[np.array((0,0,0))] for k in range(args.window_before)] # placeholder: first window values are 0 they will be deleted anyway 
for i in range(args.window_before,len(data)):
    j=1
    while j < args.window_before:
        if (abs(data[i][1]-data[i-j][1]) > 1e-6) or (abs(data[i][2]-data[i-j][2]) > 1e-6) or (abs(data[i][3]-data[i-j][3]) > 1e-6) :
            last_position.append([data[i-j,1:4]])
            break
        j+=1
    if j == args.window_before:
        last_position.append([data[i-j,1:4]])
last_position = np.array(last_position).squeeze()

next_position = []
for i in range(0,len(data)-args.window_after):
    j=1
    while j < args.window_after:
        if (abs(data[i][1]-data[i+j][1]) > 1e-6) or (abs(data[i][2]-data[i+j][2]) > 1e-6) or (abs(data[i][3]-data[i+j][3]) > 1e-6) :
            next_position.append([data[i+j,1:4]])
            break
        j+=1
    if j == args.window_after:
        next_position.append([data[i+j,1:4]])
# placeholder: last window values are 0 they will be deleted anyway 
for k in range(args.window_after):
    next_position.append([np.array((0,0,0))])
next_position = np.array(next_position).squeeze()


data_in = np.hstack((data[:,1:4],)) # first position
if args.last_pos:
    data_in = np.hstack((data_in, last_position))
if args.next_pos:
    data_in = np.hstack((data_in, next_position))

if args.specific:
    data_in = np.hstack((
        data_in,
        data[:,4:7],
    ))  # pos(t) , forces (t) , forces (t+10) , forces (t-10)
    if args.window_after>0:
        data_in = np.hstack((
            data_in,
            np.roll(data[:,4:7], -args.window_after, axis=0),
        ))
    if args.window_before>0:
        data_in = np.hstack((
            data_in,
            np.roll(data[:,4:7], +args.window_before, axis=0),
        ))

        if args.add_five:
            data_in = np.hstack((
                    data_in,
                    np.roll(data[:,4:7], -5, axis=0),
                ))
            data_in = np.hstack((
                    data_in,
                    np.roll(data[:,4:7], +5, axis=0),
                ))
else:
    data_in = np.hstack((
        data_in,
        *[np.roll(data[:,4:7], -i, axis=0) for i in range(-args.window_before, args.window_after+1)],
    ))  # pos(t) , forces (t-10) ,... forces (t-1), forces (t)   # 10 windows of forces


data_out = data[:,data_columns]


#correct the first and last 10 samples

window_before=10
window_after=10

if window_after>0 and window_before>0:
    data_in = data_in[window_before:-window_after]   #  -+10 to have the same size as ruppel
    data_out = data_out[window_before:-window_after] 
elif window_after>0 and window_before==0:
    data_in = data_in[:-window_after]
    data_out = data_out[:-window_after]
elif window_after==0 and window_before>0:
    data_in = data_in[window_before:]
    data_out = data_out[window_before:]

in_cols = data_in.shape[1]
out_cols = data_out.shape[1]

#################
# split data in train and test set:

folds=10
test_size = 1000
nb_test = 30

train_data_in_folds, train_data_out_folds, train_data_in_scaled_folds, train_data_out_scaled_folds, test_data_in_folds, test_data_out_folds, test_data_in_scaled_folds, test_data_out_scaled_folds, mean_train_in_folds, std_train_in_folds, mean_train_out_folds, std_train_out_folds = [], [], [], [], [], [], [], [], [], [], [], [] 

data_splits_indexes = np.arange(data_in.shape[0]// test_size)
np.random.shuffle(data_splits_indexes) 

for j in range(folds):
    if j<folds-1:
        idx_test=np.sort(data_splits_indexes[j*nb_test:(j+1)*nb_test]*test_size)   # take 30 chunks for test
    else:
        idx_test=np.sort(data_splits_indexes[j*nb_test:]*test_size)  # the last one do have 22 chunks

    # make sure to delete the window around the test samples
    test_samples_indexes=[]
    for i in idx_test:
        chunk=list(range(i, i+test_size))
        if (i//test_size)==len(data_splits_indexes):
            chunk=list(range(i, len(data_in)))
        if (((i//test_size)-1) not in idx_test) and  ((i//test_size)!=0):
            if window_before>0:
                chunk=chunk[window_before:]
            
        if (((i//test_size)+1) not in idx_test) and ((i//test_size)!=len(data_splits_indexes)):
            if window_after>0:
                chunk=chunk[:-window_after]
        test_samples_indexes=np.concatenate((test_samples_indexes,chunk))
    test_samples_indexes=test_samples_indexes.astype(int)
    idx_train = ((np.setdiff1d(data_splits_indexes, idx_test//test_size))*test_size).astype(int)
    
    train_samples_indexes=[]
    for i in idx_train:
        chunk=list(range(i, i+test_size))
        if (i//test_size)==len(data_splits_indexes):
            chunk=list(range(i, len(data_in)))
        if (((i//test_size)-1) not in idx_train) and  ((i//test_size)!=0):
            if window_before>0:
                chunk=chunk[window_before:]
        if (((i//test_size)+1) not in idx_train) and ((i//test_size)!=len(data_splits_indexes)):
            if window_after>0:
                chunk=chunk[:-window_after]
        train_samples_indexes=np.concatenate((train_samples_indexes,chunk))
    train_samples_indexes=train_samples_indexes.astype(int)


    # split data into training and test
    data_in_train = data_in[train_samples_indexes]
    data_out_train = data_out[train_samples_indexes]
    data_in_test = data_in[test_samples_indexes]
    data_out_test = data_out[test_samples_indexes]

    #NOTE: mean and std from all data(train + test)
    mean_in = np.mean(np.concatenate((data_in_train,data_in_test)), axis=0, keepdims=True)
    mean_out = np.mean(np.concatenate((data_out_train,data_out_test)), axis=0, keepdims=True)
    std_in = np.std(np.concatenate((data_in_train,data_in_test)), axis=0, keepdims=True)
    std_out = np.std(np.concatenate((data_out_train,data_out_test)), axis=0, keepdims=True)

    data_in_train_scaled = (data_in_train - mean_in ) / std_in
    data_out_train_scaled = (data_out_train - mean_out) / std_out
    data_in_test_scaled = (data_in_test - mean_in) / std_in
    data_out_test_scaled = (data_out_test - mean_out) / std_out

    train_data_in_folds.append(data_in_train)
    train_data_out_folds.append(data_out_train)
    train_data_in_scaled_folds.append(data_in_train_scaled)
    train_data_out_scaled_folds.append(data_out_train_scaled)
    test_data_in_folds.append(data_in_test)
    test_data_out_folds.append(data_out_test)
    test_data_in_scaled_folds.append(data_in_test_scaled)
    test_data_out_scaled_folds.append(data_out_test_scaled)
    mean_train_in_folds.append(mean_in)
    std_train_in_folds.append(std_in)
    mean_train_out_folds.append(mean_out)
    std_train_out_folds.append(std_out)

class dataset(Dataset):
    def __init__(self, data_in, data_out):
        self.data_in = data_in.astype(np.float32)
        self.data_out = data_out.astype(np.float32)

    def __len__(self):
        return len(self.data_out)

    def __getitem__(self, idx):
        # tensor 
        x = torch.from_numpy(self.data_in[idx])
        y = torch.from_numpy(self.data_out[idx])
        return x,y

class Network_FFNN(pl.LightningModule):
    def __init__(self, in_cols, out_cols,fold_ind, config):
        super(Network_FFNN, self).__init__()
        self.num_layers = config["n_layer"]
        self.linear_layers = self.build_layers(n_layer=config["n_layer"], in_features=in_cols,out_features=out_cols, n_neurons=[config["n_neurons_"+str(k)] for k in range(1,self.num_layers+1)], activations=[config["activation_"+str(k)] for k in range(1,self.num_layers+1)], neg_slope_leakyrelu=config["neg_slope_leakyrelu"])
        self.config = config
        self.fold = fold_ind
        self.save_hyperparameters()

    def build_layers(self, n_layer, in_features, out_features, n_neurons, activations, neg_slope_leakyrelu):
        layers = []
        for ind in range(n_layer):
            if ind == 0:
                layers.append(nn.Linear(in_features, n_neurons[0]))
            elif ind == (n_layer-1):
                layers.append(nn.Linear(n_neurons[ind-1], out_features))
            else:
                layers.append(nn.Linear(n_neurons[ind-1], n_neurons[ind]))

            if activations[ind] == 'relu':
                layers.append(nn.ReLU())
            elif activations[ind] == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activations[ind] == 'hardtanh':
                layers.append(nn.Hardtanh())
            elif activations[ind] == 'tanh':
                layers.append(nn.Tanh())
            elif activations[ind] == 'elu':
                layers.append(nn.ELU())
            elif activations[ind] == 'leakyrelu':
                layers.append(nn.LeakyReLU(negative_slope=neg_slope_leakyrelu))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        l = self.linear_layers(x)
        return l

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["learning_rate_init"],)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.custom_loss(y, y_pred)
        self.log('train_loss_'+str(self.fold), loss)
        smae=self.smae(y, y_pred)
        self.log('train_smae_'+str(self.fold), smae)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        return y_pred
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.custom_loss(y, y_pred)
        self.log('val_loss_'+str(self.fold), loss)
        smae=self.smae(y, y_pred)
        self.log('val_smae_'+str(self.fold), smae)
        return y_pred


    def custom_loss(self, y_true, y_pred):
        err = y_pred - y_true
        err = (torch.abs(err)) + torch.pow(err, 2)
        return torch.mean(err)

    def smae(self, y_true, y_pred):
        err = y_pred - y_true
        err = (torch.abs(err)) 
        return torch.mean(err)




def measure_inference_usage(model, test_inputs):
    x, y = next(iter(test_inputs))
    inference=[]
    # warmup
    model.eval()
    for i in range(10):
        _ = model(x) 
    for i in range(100):
        start = time.time()
        _ = model(x)
        end = time.time()
        time_elapsed = end - start
        time_elapsed = time_elapsed * 1000 # ms
        inference.append(time_elapsed)
    return np.mean(inference), np.std(inference)


# Cross Validation
data={}

all_mae = []
all_smae = []
all_mae_electrodes = []
all_smae_electrodes = []
all_rmse = []
all_nrmse = []
all_rmse_electrodes = []
all_nrmse_electrodes = []

fold_ind=0
for train_data_in, train_data_out, train_data_in_scaled, train_data_out_scaled, test_data_in, test_data_out, test_data_in_scaled, test_data_out_scaled, mean_out, std_out  in zip(train_data_in_folds, train_data_out_folds, train_data_in_scaled_folds, train_data_out_scaled_folds, test_data_in_folds, test_data_out_folds, test_data_in_scaled_folds, test_data_out_scaled_folds, mean_train_out_folds, std_train_out_folds):
        
    print("Fold: ", fold_ind)

    data["Fold"+str(fold_ind)]={}
        

    # split train data into train and validation
    nb_val = 30
    val_size = 1000
    # select 30 random numbers between 0+val_size and 30000-val_size
    idx_val = np.random.randint(val_size, train_data_in.shape[0]-val_size, size=nb_val)
    idx_val = np.sort(idx_val)
    while (np.any(np.diff(idx_val) <= val_size)):
        idx_val = np.random.randint(val_size, train_data_in.shape[0]-val_size, size=nb_val)
        idx_val = np.sort(idx_val)
    
    val_samples_indexes=[]
    for i in idx_val:
        chunk=list(range(i+window_before, i+val_size-window_after))
        val_samples_indexes=np.concatenate((val_samples_indexes,chunk))
    val_samples_indexes=val_samples_indexes.astype(int)

    idx_train = idx_val + val_size

    train_samples_indexes=list(range(0, idx_train[0] - val_size-window_after)) # i am sure that the first chunk is not in the validation set
    for i in range(0,len(idx_train)-1):
        chunk=list(range(idx_train[i]+window_before, idx_train[i+1]-val_size-window_after))
        train_samples_indexes=np.concatenate((train_samples_indexes,chunk))
    # add the last chunk
    train_samples_indexes=np.concatenate((train_samples_indexes,list(range(idx_train[-1]+window_before, train_data_in.shape[0]))))
    # int
    train_samples_indexes=train_samples_indexes.astype(int)

    # split data into training and validation
    train_data_in_train = train_data_in[train_samples_indexes]
    train_data_in_train_scaled = train_data_in_scaled[train_samples_indexes]
    train_data_out_train = train_data_out[train_samples_indexes]
    train_data_out_train_scaled = train_data_out_scaled[train_samples_indexes]
    train_data_in_val = train_data_in[val_samples_indexes]
    train_data_in_val_scaled = train_data_in_scaled[val_samples_indexes]
    train_data_out_val = train_data_out[val_samples_indexes]
    train_data_out_val_scaled = train_data_out_scaled[val_samples_indexes]


    train_dataset = dataset(train_data_in_train_scaled, train_data_out_train_scaled)
    validation_dataset = dataset(train_data_in_val_scaled, train_data_out_val_scaled)
    test_dataset = dataset(test_data_in_scaled, test_data_out_scaled)

    train_loader = DataLoader(train_dataset, batch_size=Best_config["batch_size"], shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=1024, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4)

    model = Network_FFNN(in_cols, out_cols, fold_ind, Best_config)
    #print(model)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd()+"/checkpoints/"+wandb_logger.experiment.name+"_Fold"+str(fold_ind)+"/",
        save_top_k=1,
        verbose=True,
        monitor='val_loss_'+str(fold_ind),
        mode='min',
    )

    trainer = pl.Trainer(accelerator="cpu", max_epochs=100, deterministic=True,logger=wandb_logger,callbacks=[utils.MyProgressBar(),checkpoint_callback,EarlyStopping(monitor="val_loss_"+str(fold_ind), mode="min", patience=8)],enable_checkpointing=True)
    
    
    #TODO:
    #trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    #TODO:
    #model = Network_FFNN.load_from_checkpoint(checkpoint_callback.best_model_path, in_cols=in_cols, out_cols=out_cols, config=Best_config)
    #predictions = trainer.predict(model=model, dataloaders=test_loader,)

    if fold_ind==0:
        #get one sample
        test_dataset_one_input = dataset( test_data_in_scaled[0:1], test_data_out_scaled[0:1])
        test_loader_one_input = DataLoader(test_dataset_one_input, batch_size=1, shuffle=False, num_workers=4)    
        inference_mean, inference_std = measure_inference_usage(model, test_loader_one_input,)
        wandb.log({"inference_time": inference_mean})
        wandb.log({"inference_time_std": inference_std})
    #TODO:
    break


wandb.finish()