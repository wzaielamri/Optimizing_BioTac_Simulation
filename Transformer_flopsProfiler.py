#!/usr/bin/env python

Best_config ={
    "num_layers": 7,
    "embed_dim": 64,
    "hidden_dim": 128,
    "num_heads": 1,
    "dropout": 0.0,
    "learning_rate": 0.0004,
    "batch_size": 256,
}

import numpy as np
import csv
import os
import sys
import logging
import utils
from os.path import basename
import psutil


from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import wandb

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch.optim as optim
from torch.utils.data import Dataset
import time
import argparse

from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator


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

exp_name="Transformer_SMACBest_Pytorch"
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
        "note": "SMACBest Setting: TrainValTest dataset Split Pytorch EralyStop Transformers "+str(args.prefix),
        "optimizer": "Adam",
        "loss": "default_loss",
    })

if args.load_smac:
    smac_path="smac3_output/SMACSearch_Transformer_pytorch"
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
    Best_config=utils.BestSMAC_Transformer(smac_path)

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

# code adjusted from: pytorch lightning tutorial on ViT 
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """Attention Block.

        Args:
            embed_dim: Dimensionality of input and attention feature vectors
            hidden_dim: Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads: Number of heads to use in the Multi-Head Attention block
            dropout: Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class NetworkTransformer(nn.Module):
    def __init__(
        self,model_args
    ):
        """Vision Transformer.

        Args:
            embed_dim: Dimensionality of the input feature vectors to the Transformer
            hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_heads: Number of heads to use in the Multi-Head Attention block
            num_layers: Number of layers to use in the Transformer
            num_outputs: Number of outputs to predict
            patch_size: Number of pixels that the patches have per dimension
            num_patches: Maximum number of patches an image can have
            dropout: Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = model_args["patch_size"]
        self.num_patches =  model_args["num_patches"]
        self.embed_dim = model_args["embed_dim"]
        self.hidden_dim = model_args["hidden_dim"]
        self.num_heads = model_args["num_heads"]
        self.num_layers = model_args["num_layers"]
        self.num_outputs = model_args["num_outputs"]
        self.dropout = model_args["dropout"]

        # Layers/Networks
        self.input_layer = nn.Linear(self.patch_size, self.embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(self.embed_dim, self.hidden_dim, self.num_heads, dropout=self.dropout) for _ in range(self.num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, self.num_outputs))
        self.dropout = nn.Dropout(self.dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + self.num_patches, self.embed_dim))

    def forward(self, x):
        # Preprocess input
        x=x.reshape((x.shape[0], self.num_patches, self.patch_size))

        B, T, _ = x.shape
        x = self.input_layer(x)
        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out


class Network_Transformer(pl.LightningModule):
    def __init__(self, model_args,):
        super(Network_Transformer, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = model_args["learning_rate"]
        self.model = NetworkTransformer(model_args)
        self.fold = model_args["fold"]
    def forward(self, x):
        l = self.model(x)
        return l

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

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



model_args={"embed_dim": Best_config["embed_dim"],
    "hidden_dim": Best_config["hidden_dim"],
    "num_heads": Best_config["num_heads"],
    "num_layers": Best_config["num_layers"],
    "patch_size": 3,
    "num_patches": in_cols//3, 
    "num_outputs": out_cols,
    "dropout": Best_config["dropout"],
    "learning_rate": Best_config["learning_rate"],
    "fold": 0,}

model = Network_Transformer(model_args=model_args,)
#print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


trainer = pl.Trainer(accelerator="gpu", devices=1,max_epochs=100, deterministic=True,logger=wandb_logger,callbacks=[utils.MyProgressBar(), EarlyStopping(monitor="val_loss_"+str(0), mode="min", patience=8),] #LearningRateMonitor("epoch")
                        ,enable_checkpointing=True)

flops, macs, params = flops_profiler(model, in_cols, output_file="./flops_profiler/"+str(wandb_logger.experiment.name)+".txt")
wandb.log({"flops": flops})
wandb.log({"macs": macs})
wandb.log({"params": params})
    
wandb.finish()