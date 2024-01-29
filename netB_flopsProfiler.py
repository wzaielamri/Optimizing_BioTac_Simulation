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

from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator

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


name=name+"_FlopsProfiler"



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


def flops_profiler(model, in_cols, output_file):

    with get_accelerator().device(0):
        batch_size = 1
        flops, macs, params = get_model_profile(model=model, # model
                                        input_shape=(batch_size, in_cols), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                        args=None, # list of positional arguments to the model.
                                        kwargs=None, # dictionary of keyword arguments to the model.
                                        print_profile=True, # prints the model graph with the measured profile attached to each module
                                        detailed=True, # print the detailed profile
                                        module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                        top_modules=1, # the number of top modules to print aggregated profile
                                        warm_up=10, # the number of warm-ups before measuring the time of each module
                                        as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                        output_file=output_file, # path to the output file. If None, the profiler prints to stdout.
                                        ignore_modules=None) # the list of modules to ignore in the profiling
    return flops, macs, params



model = NetworkB(in_cols, out_cols, 0)

# train model
trainer = pl.Trainer(accelerator="gpu", devices=1, logger=wandb_logger, max_epochs=50, deterministic=True, callbacks=[utils.MyProgressBar()])

wandb_logger.experiment.config.update({"epoch": 50}) # just to init the logging

flops, macs, params = flops_profiler(model, in_cols, output_file="./flops_profiler/"+str(wandb_logger.experiment.name)+".txt")
wandb.log({"flops": flops})
wandb.log({"macs": macs})
wandb.log({"params": params})

wandb.finish()