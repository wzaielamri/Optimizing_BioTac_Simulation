# to split test also :

#!/usr/bin/env python

import numpy as np
import csv
import os
import sys
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from os.path import basename
import torch.optim as optim
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset
import utils
import wandb
import time
import psutil
from fvcore.nn.flop_count import flop_count


np.random.seed(42)
pl.seed_everything(42, workers=True)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--window_before', type=int, default=10, help='window before')
parser.add_argument('--window_after', type=int, default=10, help='window after')
parser.add_argument('--specific', type=bool, default=False, help='specific')
parser.add_argument('--next_pos', type=bool, default=False, help='next_pos')
parser.add_argument('--last_pos', type=bool, default=False, help='last_pos')
parser.add_argument('--add_five', type=bool, default=False, help='add_five')
parser.add_argument("--file", type=str, default="./biotac_single_contact_response/2018-01-19-18-16-58_biotac_ff_stick_calibration.bag.csv", help="file path")



args = parser.parse_args()

if args.specific:
    if not args.add_five:
        name="NetworkB_Pytorch_DefaultConfig_NoWindow_-"+str(args.window_before)+"+"+str(args.window_after)
    else:
        name="NetworkB_Pytorch_DefaultConfig_NoWindow_-"+str(args.window_before)+"+"+str(args.window_after)+"-5+5"

else:
    name="NetworkB_Pytorch_DefaultConfig_Window_-"+str(args.window_before)+"+"+str(args.window_after)

if args.last_pos:
    name=name+"_lastPos"
if args.next_pos:
    name=name+"_nextPos"


name=name+"_CV_testvaltrain"



wandb_logger = WandbLogger(log_model="all",    
project="BioTacPlugin",
name=name,
# run_name
# track hyperparameters and run metadata with wandb.config
config={
    "note": "Default Setting: TrainVal dataset Split NetworkB Pytorch",
    "optimizer": "Adam",
    "loss": "default_loss",
    "epoch": 50,
    "batch_size": 1024
})


file_path = args.file

reader = csv.reader(open(file_path))

rows = [row for row in reader]


data_columns = [7, 8, 9, 10] + list(range(12, 31))

headers_out = np.array(rows[0])[data_columns].tolist()

rows = rows[1:]
data = np.array(rows).astype(float)



last_position = [[np.array((0,0,0))] for k in range(10)] # placeholder: first window values are 0 they will be deleted anyway 
for i in range(10,len(data)):
    j=1
    while j < 10:
        if (abs(data[i][1]-data[i-j][1]) > 1e-6) or (abs(data[i][2]-data[i-j][2]) > 1e-6) or (abs(data[i][3]-data[i-j][3]) > 1e-6) :
            last_position.append([data[i-j,1:4]])
            break
        j+=1
    if j == 10:
        last_position.append([data[i-j,1:4]])
last_position = np.array(last_position).squeeze()

next_position = []
for i in range(0,len(data)-10):
    j=1
    while j < 10:
        if (abs(data[i][1]-data[i+j][1]) > 1e-6) or (abs(data[i][2]-data[i+j][2]) > 1e-6) or (abs(data[i][3]-data[i+j][3]) > 1e-6) :
            next_position.append([data[i+j,1:4]])
            break
        j+=1
    if j == 10:
        next_position.append([data[i+j,1:4]])
# placeholder: last window values are 0 they will be deleted anyway 
for k in range(10):
    next_position.append([np.array((0,0,0))])
next_position = np.array(next_position).squeeze()


data_in = np.hstack((
    data[:,1:4],
))

if args.next_pos and args.next_pos:
    data_in = np.hstack((
        data_in,
        last_position,
        next_position,
    ))

if args.specific:
    data_in = np.hstack((
        data_in,
        data[:,4:7],
    ))
    if args.window_before>0:
        data_in = np.hstack((
            data_in,
            np.roll(data[:,4:7], +args.window_before, axis=0), # future
        ))
    if args.window_after>0:
        data_in = np.hstack((
            data_in,
            np.roll(data[:,4:7], -args.window_after, axis=0), # past
        ))
    if args.add_five:
        data_in = np.hstack((
            data_in,
            np.roll(data[:,4:7], +5, axis=0), # future
            np.roll(data[:,4:7], -5, axis=0), # past
        ))
else:
    if args.window_before>0:
        data_in = np.hstack((
            data_in,
            *[np.roll(data[:,4:7], -i, axis=0) for i in range(-args.window_before, args.window_after+1)],
        ))


data_in = np.hstack((
    data_in,
    data[:,11:12],
))


if args.last_pos and args.next_pos:
    range_pos = list(range(0,9))
else:
    range_pos = list(range(0,3))

if args.specific:
    if args.add_five:
        range_force = list(range(len(range_pos),(len(range_pos)+15)))
    else:
        if args.window_after>0:
            range_force = list(range(len(range_pos),len(range_pos)+3*3))
        else:
            range_force = list(range(len(range_pos),len(range_pos)+3*2))
        if args.window_before==0:
            range_force = list(range(len(range_pos),len(range_pos)+3))
else:
    if args.window_after>0:
        range_force = list(range(len(range_pos),len(range_pos)+3*21))
    else:
        range_force = list(range(len(range_pos),len(range_pos)+3*11))


range_temp = list(range(len(range_pos)+len(range_force),len(range_pos)+len(range_force)+1))

data_out = data[:,data_columns]


headers_in = [
    "px", "py", "pz",
    "fx", "fy", "fz", "fx1", "fy1", "fz1", "fx2", "fy2", "fz2", "t"
    ]
#correct the first 10 samples
data_in = data_in[10:-10]
data_out = data_out[10:-10]

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
            chunk=chunk[10:]
        if (((i//test_size)+1) not in idx_test) and ((i//test_size)!=len(data_splits_indexes)):
            chunk=chunk[:-10]
        test_samples_indexes=np.concatenate((test_samples_indexes,chunk))
    test_samples_indexes=test_samples_indexes.astype(int)
    idx_train = ((np.setdiff1d(data_splits_indexes, idx_test//test_size))*test_size).astype(int)
    
    train_samples_indexes=[]
    for i in idx_train:
        chunk=list(range(i, i+test_size))
        if (i//test_size)==len(data_splits_indexes):
            chunk=list(range(i, len(data_in)))
        if (((i//test_size)-1) not in idx_train) and  ((i//test_size)!=0):
            chunk=chunk[10:]
        if (((i//test_size)+1) not in idx_train) and ((i//test_size)!=len(data_splits_indexes)):
            chunk=chunk[:-10]
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

#################


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
    
#################


class NetworkB(pl.LightningModule):
    def __init__(self, in_cols, out_cols, fold_ind):
        super(NetworkB, self).__init__()
        self.fold_ind = fold_ind
        self.area_selector = self.Selector(range_pos,  input_shape=(in_cols,))
        self.force_selector = self.Selector(range_force,  input_shape=(in_cols,))
        self.temperature_selector = self.Selector(range_temp,  input_shape=(in_cols,))
        
        self.area_layers = self.build_layers( len(range_pos), [512,512,512,64], activations=['relu','relu','relu','linear'])
        self.force_layers = self.build_layers(len(range_force), [256,256,256,64], activations=['relu','relu','relu','linear']) #, bias_regularizer=nn.L1Loss(0.0015))
        self.activation_layers = self.build_layers(64, [256,256,23], activations=['relu','relu','linear'])
        self.temperature_layers = self.build_layers( 1, [256,23], activations=['sigmoid','linear'])
        self.linear_layers = self.build_layers(23, [out_cols], activations=['linear'])


    def Selector(self, indices, input_shape):
        weights = torch.zeros(len(indices), input_shape[0])
        for i in range(len(indices)):
            weights[i, indices[i]] = 1
        layer = nn.Linear(input_shape[0], len(indices), bias=False)
        with torch.no_grad():
            layer.weight.copy_(weights)
        layer.weight.requires_grad = False
        return layer

    def build_layers(self, in_features , neuron_layers, activations, bias_regularizer=None):
        layers = []
        layers.append(nn.Linear(in_features, neuron_layers[0]))
        if activations[0] == 'relu':
            layers.append(nn.ReLU())
        elif activations[0] == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activations[0] == 'linear':
            layers.append(nn.Identity())
        if bias_regularizer:
            layers.append(bias_regularizer)

        for ind, nb_neuron in enumerate(neuron_layers[1:]):
            layers.append(nn.Linear(neuron_layers[ind], nb_neuron))
            if activations[ind+1] == 'relu':
                layers.append(nn.ReLU())
            elif activations[ind+1] == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activations[ind+1] == 'linear':
                layers.append(nn.Identity())
            if bias_regularizer:
                layers.append(bias_regularizer)
        return nn.Sequential(*layers)

    def forward(self, x):
        area = self.area_selector(x)
        force = self.force_selector(x)
        temperature = self.temperature_selector(x)
        area = self.area_layers(area)
        force = self.force_layers(force)
        activation = force * area
        activation = self.activation_layers(activation)
        temperature = self.temperature_layers(temperature)
        l = activation + temperature
        l = self.linear_layers(l)
        return l

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.00005, betas=(0.97, 0.999))
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.custom_loss(y, y_pred)
        self.log('train_loss_'+str(self.fold_ind), loss)
        smae=self.smae(y, y_pred)
        self.log('train_smae_'+str(self.fold_ind), smae)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.custom_loss(y, y_pred)
        self.log('val_loss_'+str(self.fold_ind), loss)
        smae=self.smae(y, y_pred)
        self.log('val_smae_'+str(self.fold_ind), smae)
        return y_pred

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        return y_pred
    def custom_loss(self, y_true, y_pred):
        err = y_pred - y_true
        err = torch.abs(err) + torch.pow(err, 2)
        return torch.mean(err)

    def smae(self, y_true, y_pred):
        err = y_pred[:,[0,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]] - y_true[:,[0,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]
        err = (torch.abs(err)) 
        return torch.mean(err)
###############



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
all_mae_channels = []
all_smae_channels = []
all_mape_channels = []
all_rmse_channels = []
all_best_temps = []
all_best_temps_scaled = []

fold_ind=0


for train_data_in, train_data_out, train_data_in_scaled, train_data_out_scaled, test_data_in, test_data_out, test_data_in_scaled, test_data_out_scaled, mean_out, std_out  in zip(train_data_in_folds, train_data_out_folds, train_data_in_scaled_folds, train_data_out_scaled_folds, test_data_in_folds, test_data_out_folds, test_data_in_scaled_folds, test_data_out_scaled_folds, mean_train_out_folds, std_train_out_folds):
        
    print("Fold: ", fold_ind)


    # split train data into train and validation
    nb_val = 30
    val_size = 1000
    # select 30 random numbers between 0 and 30000
    idx_val = np.random.randint(val_size, train_data_in.shape[0]-val_size, size=nb_val)
    idx_val = np.sort(idx_val)
    while (np.any(np.diff(idx_val) <= val_size)):
        idx_val = np.random.randint(val_size, train_data_in.shape[0]-val_size, size=nb_val)
        idx_val = np.sort(idx_val)
    
    val_samples_indexes=[]
    for i in idx_val:
        chunk=list(range(i+10, i+val_size-10))
        val_samples_indexes=np.concatenate((val_samples_indexes,chunk))
    val_samples_indexes=val_samples_indexes.astype(int)

    idx_train = idx_val + val_size

    train_samples_indexes=list(range(0, idx_train[0] - val_size-10)) # i am sure that the first chunk is not in the validation set
    for i in range(0,len(idx_train)-1):
        chunk=list(range(idx_train[i]+10, idx_train[i+1]-val_size-10))
        train_samples_indexes=np.concatenate((train_samples_indexes,chunk))
    # add the last chunk
    train_samples_indexes=np.concatenate((train_samples_indexes,list(range(idx_train[-1]+10, train_data_in.shape[0]))))
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

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=1024, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4)

    model = NetworkB(in_cols, out_cols, fold_ind)

    # train model
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=wandb_logger, max_epochs=50, deterministic=True, callbacks=[utils.MyProgressBar()])

    wandb_logger.experiment.config.update({"epoch": 50}) # just to init the logging

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    # first predict with the correct temperature
    predictions = trainer.predict(model=model, dataloaders=test_loader,)

    
    # save model checkpoint
    trainer.save_checkpoint('./checkpoints/'+str(wandb_logger.experiment.name)+'_fold_'+str(fold_ind)+'.ckpt')

    if fold_ind==0:
        test_dataset_one_input = dataset( test_data_in_scaled[0:1], test_data_out_scaled[0:1])
        test_loader_one_input = DataLoader(test_dataset_one_input, batch_size=1, shuffle=False, num_workers=4)    


    predictions = np.vstack(predictions)
    y_pred = predictions * std_out + mean_out

    mae = np.mean(np.abs(y_pred - test_data_out), axis=0)
    #std_train_channels = np.std(train_data_out, axis=0)
    #smae = mae / std_out
    smae = np.mean(np.abs(predictions - test_data_out_scaled), axis=0)

    mape = np.mean(np.abs((y_pred - test_data_out) / test_data_out), axis=0)
    rmse = np.sqrt(np.mean(np.power((y_pred - test_data_out),2), axis=0))
    nrmse = np.sqrt(np.mean(np.power((predictions - test_data_out_scaled),2), axis=0))
    mae_electrodes = np.mean(mae[-19:])
    smae_electrodes = np.mean(smae[-19:])

    # save data into dictionary
    data["Fold"+str(fold_ind)]={}
    data["Fold"+str(fold_ind)]["mae_correct_temp"]=mae
    data["Fold"+str(fold_ind)]["smae_correct_temp"]=smae
    data["Fold"+str(fold_ind)]["mape_correct_temp"]=mape
    data["Fold"+str(fold_ind)]["rmse_correct_temp"]=rmse
    data["Fold"+str(fold_ind)]["nrmse_correct_temp"]=nrmse


    # choose best temperature
    temp=np.concatenate((train_data_in[:,-1],test_data_in[:,-1])) #train_data_in[:,-1]
    temp_scaled=np.concatenate((train_data_in_scaled[:,-1],test_data_in_scaled[:,-1])) #train_data_in_scaled[:,-1]
    test_data_in_scaled_fixed_temp= test_data_in_scaled.copy()
    list_temperature_scaled=np.unique(np.sort(temp_scaled)) # we take all temperatures in the dataset and test them
    list_temperature=np.unique(np.sort(temp))
    list_mae_all_temperature=[]
    list_smae_all_temperature=[]
    list_mae_electrodes_all_temperature=[]
    list_smae_electrodes_all_temperature=[]
    list_mae_channels_all_temperature=[]
    list_smae_channels_all_temperature=[]
    list_mape_channels_all_temperature=[]
    list_rmse_channels_all_temperature=[]
    list_nrmse_channels_all_temperature=[]

    for temp in list_temperature_scaled:
        test_data_in_scaled_fixed_temp[:,-1]=temp
        test_dataset_fixed_temp = dataset(test_data_in_scaled_fixed_temp, test_data_out_scaled)
        test_loader_fixed_temp = DataLoader(test_dataset_fixed_temp, batch_size=1024, shuffle=False, num_workers=4)
        # predict on test set
        predictions = trainer.predict(model=model, dataloaders=test_loader_fixed_temp,)

        predictions = np.vstack(predictions)
        y_pred = predictions * std_out + mean_out


        mae = np.mean(np.abs(y_pred - test_data_out), axis=0)
        #std_train_channels = np.std(train_data_out, axis=0)
        #smae = mae / std_out
        smae = np.mean(np.abs(predictions - test_data_out_scaled), axis=0)

        mape = np.mean(np.abs((y_pred - test_data_out) / test_data_out), axis=0)
        rmse = np.sqrt(np.mean(np.power((y_pred - test_data_out),2), axis=0))
        nrmse =np.sqrt(np.mean(np.power((predictions - test_data_out_scaled),2), axis=0))
        list_mae_all_temperature.append(np.mean((np.concatenate(([mae[0]],[mae[2]],mae[-19:]))))) # pac_0, pdc, electrodes same as in the paper
        list_smae_all_temperature.append(np.mean((np.concatenate(([smae[0]],[smae[2]],smae[-19:])))))# pac_0, pdc, electrodes same as in the paper
        list_mae_electrodes_all_temperature.append(np.mean(mae[-19:]))
        list_smae_electrodes_all_temperature.append(np.mean(smae[-19:]))
        list_mae_channels_all_temperature.append(mae)
        list_smae_channels_all_temperature.append(smae)
        list_mape_channels_all_temperature.append(mape)
        list_rmse_channels_all_temperature.append(rmse)
        list_nrmse_channels_all_temperature.append(nrmse)




    # after finding the lowest smae, we choose the corresponding temperature
    best_temp_scaled=list_temperature_scaled[np.argmin(list_smae_all_temperature)]
    best_temp=list_temperature[np.argmin(list_smae_all_temperature)]


    data["Fold"+str(fold_ind)]["list_temperature_scaled"]=list_temperature_scaled
    data["Fold"+str(fold_ind)]["list_temperature"]=list_temperature
    data["Fold"+str(fold_ind)]["list_mae_channels_all_temperature"]=list_mae_channels_all_temperature
    data["Fold"+str(fold_ind)]["list_smae_channels_all_temperature"]=list_smae_channels_all_temperature
    data["Fold"+str(fold_ind)]["list_mape_channels_all_temperature"]=list_mape_channels_all_temperature
    data["Fold"+str(fold_ind)]["list_rmse_channels_all_temperature"]=list_rmse_channels_all_temperature
    data["Fold"+str(fold_ind)]["list_nrmse_channels_all_temperature"]=list_nrmse_channels_all_temperature


    test_data_in_scaled_fixed_temp[:,-1]=best_temp_scaled
    test_dataset_fixed_temp = dataset(test_data_in_scaled_fixed_temp, test_data_out_scaled)
    test_loader_fixed_temp = DataLoader(test_dataset_fixed_temp, batch_size=1024, shuffle=False, num_workers=4)
    # predict on test set

    predictions = trainer.predict(model=model, dataloaders=test_loader_fixed_temp,)
    

    predictions = np.vstack(predictions)
    y_pred = predictions * std_out + mean_out


    if fold_ind==0:
        np.save('./results_cv/'+str(wandb_logger.experiment.name)+'_fold_0_predictions.npy', y_pred)
        np.save('./results_cv/'+str(wandb_logger.experiment.name)+'_fold_0_test_data_out.npy', test_data_out)


    mae = np.mean(np.abs(y_pred - test_data_out), axis=0)
    #std_train_channels = np.std(train_data_out, axis=0)
    #smae = mae / std_out
    smae = np.mean(np.abs(predictions - test_data_out_scaled), axis=0)

    mape = np.mean(np.abs((y_pred - test_data_out) / test_data_out), axis=0)
    rmse = np.sqrt(np.mean(np.power((y_pred - test_data_out),2), axis=0))
    nrmse =np.sqrt(np.mean(np.power((predictions - test_data_out_scaled),2), axis=0))

    mae_electrodes = np.mean(mae[-19:])
    smae_electrodes = np.mean(smae[-19:])
    rmse_electrodes = np.mean(rmse[-19:])
    nrmse_electrodes = np.mean(nrmse[-19:])

    print("Mean test_mae_electrodes: ", mae_electrodes)
    print("Mean test_mae_scaled_electrodes: ", smae_electrodes)


    all_mae_channels.append(mae)
    all_smae_channels.append(smae)
    all_mape_channels.append(mape)
    all_rmse_channels.append(rmse)
    all_mae_electrodes.append(mae_electrodes)
    all_smae_electrodes.append(smae_electrodes)
    all_mae.append(np.mean((np.concatenate(([mae[0]],[mae[2]],mae[-19:])))))
    all_smae.append(np.mean((np.concatenate(([smae[0]],[smae[2]],smae[-19:])))))

    all_rmse.append((np.mean((np.concatenate(([rmse[0]],[rmse[2]],rmse[-19:])))))) # check this if problem 
    all_nrmse.append((np.mean((np.concatenate(([nrmse[0]],[nrmse[2]],nrmse[-19:])))))) # check this if problem 
    all_rmse_electrodes.append(rmse_electrodes)
    all_nrmse_electrodes.append(nrmse_electrodes)

    all_best_temps.append(best_temp)
    all_best_temps_scaled.append(best_temp_scaled)

    # save data into dictionary
    data["Fold"+str(fold_ind)]["best_temp"]=best_temp
    data["Fold"+str(fold_ind)]["mae_channels_best_temp"]=mae
    data["Fold"+str(fold_ind)]["smae_channels_best_temp"]=smae
    data["Fold"+str(fold_ind)]["mape_channels_best_temp"]=mape
    data["Fold"+str(fold_ind)]["rmse_channels_best_temp"]=rmse
    data["Fold"+str(fold_ind)]["nrmse_channels_best_temp"]=nrmse

    wandb.log({"Fold"+str(fold_ind)+"_mean_smae_channels": np.mean(smae)})
    # log number of parameters
    if fold_ind==0:
        wandb.log({"nb_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

    fold_ind+=1


wandb.log({"All_Folds_mean_mae_channels": np.round(np.mean(all_mae),3)})
wandb.log({"All_Folds_std_mae_channels": np.round(np.std(all_mae),3)})
wandb.log({"All_Folds_mean_mae_electrodes": np.round(np.mean(all_mae_electrodes),3)})
wandb.log({"All_Folds_std_mae_electrodes": np.round(np.std(all_mae_electrodes),3)})

wandb.log({"All_Folds_mean_smae_channels": np.round(np.mean(all_smae),3)})
wandb.log({"All_Folds_std_smae_channels": np.round(np.std(all_smae),3)})
wandb.log({"All_Folds_mean_smae_electrodes": np.round(np.mean(all_smae_electrodes),3)})
wandb.log({"All_Folds_std_smae_electrodes": np.round(np.std(all_smae_electrodes),3)})

wandb.log({"All_Folds_mean_rmse_channels": np.round(np.mean(all_rmse),3)})
wandb.log({"All_Folds_std_rmse_channels": np.round(np.std(all_rmse),3)})
wandb.log({"All_Folds_mean_rmse_electrodes": np.round(np.mean(all_rmse_electrodes),3)})
wandb.log({"All_Folds_std_rmse_electrodes": np.round(np.std(all_rmse_electrodes),3)})

wandb.log({"All_Folds_mean_nrmse_channels": np.round(np.mean(all_nrmse),3)})
wandb.log({"All_Folds_std_nrmse_channels": np.round(np.std(all_nrmse),3)})
wandb.log({"All_Folds_mean_nrmse_electrodes": np.round(np.mean(all_nrmse_electrodes),3)})
wandb.log({"All_Folds_std_nrmse_electrodes": np.round(np.std(all_nrmse_electrodes),3)})

# save data dictionary
import pickle
with open('./results_cv/'+str(wandb_logger.experiment.name)+'_temperatures_results.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

wandb.finish()