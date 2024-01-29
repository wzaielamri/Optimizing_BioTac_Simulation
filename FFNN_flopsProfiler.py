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

from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
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
exp_name += "_FlopsProfiler"+str(args.prefix)

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
    smac_path="smac3_output/SMACSearch_FFNN_pytorch"
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


model = Network_FFNN(in_cols, out_cols, 0, Best_config)
#print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=100, deterministic=True,logger=wandb_logger,callbacks=[utils.MyProgressBar(),EarlyStopping(monitor="val_loss_"+str(0), mode="min", patience=8)],enable_checkpointing=True)


flops, macs, params = flops_profiler(model, in_cols, output_file="./flops_profiler/"+str(wandb_logger.experiment.name)+".txt")
wandb.log({"flops": flops})
wandb.log({"macs": macs})
wandb.log({"params": params})



wandb.finish()