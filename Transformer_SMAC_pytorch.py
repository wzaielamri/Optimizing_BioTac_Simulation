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
import yaml
import random
import base64
import warnings
import torch.optim as optim
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)
from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from deepcave import Recorder, Objective

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", type=str, default="", help="exp name")
parser.add_argument('--window_before', type=int, default=10, help='window before')
parser.add_argument('--window_after', type=int, default=0, help='window after')
parser.add_argument('--specific', type=bool, default=False, help='specific')
parser.add_argument('--next_pos', type=bool, default=False, help='next_pos')
parser.add_argument('--last_pos', type=bool, default=False, help='last_pos')
parser.add_argument('--num_trials', type=int, default=1000, help='number of smac trials to run')
parser.add_argument('--max_budget', type=int, default=50, help='max_budget')
parser.add_argument('--add_five', type=bool, default=False, help='add_five')
parser.add_argument("--file", type=str, default="./biotac_single_contact_response/2018-01-19-18-16-58_biotac_ff_stick_calibration.bag.csv", help="file path")


args = parser.parse_args()

exp_name="SMACSearch_Transformer_pytorch"
if args.specific:
    exp_name += "_No_Window_-"+str(args.window_before)+"+"+str(args.window_after)
else: 
    exp_name += "_Window_-"+str(args.window_before)+"+"+str(args.window_after)
if args.last_pos:
    exp_name += "_LastPos"
if args.next_pos:
    exp_name += "_NextPos"
exp_name += "_testtrain"+str(args.prefix)



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
# randomly select 30 number between 0 and 30000 and are at least 1000 steps away from each other
nb_test = 30
test_size = 1000
# select 30 random numbers between 0 and 30000
idx_test = np.random.randint(test_size, data_in.shape[0]-test_size, size=nb_test)
idx_test = np.sort(idx_test)
while (np.any(np.diff(idx_test) <= test_size)):
    idx_test = np.random.randint(test_size, data_in.shape[0]-test_size, size=nb_test)
    idx_test = np.sort(idx_test)

test_samples_indexes=[]
for i in idx_test:
    chunk=list(range(i+window_before, i+test_size-window_after))
    test_samples_indexes=np.concatenate((test_samples_indexes,chunk))
test_samples_indexes=test_samples_indexes.astype(int)

idx_train = idx_test + test_size

train_samples_indexes=list(range(0, idx_train[0] - test_size-window_after)) # i am sure that the first chunk is not in the test set
for i in range(0,len(idx_train)-1):
    chunk=list(range(idx_train[i]+window_before, idx_train[i+1]-test_size-window_after))
    train_samples_indexes=np.concatenate((train_samples_indexes,chunk))
# add the last chunk
train_samples_indexes=np.concatenate((train_samples_indexes,list(range(idx_train[-1]+window_before, data_in.shape[0]))))
# int
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

data_in_train_scaled = data_in_train.copy()
data_out_train_scaled = data_out_train.copy()
data_in_test_scaled = data_in_test.copy()
data_out_test_scaled = data_out_test.copy()

data_in_train_scaled = (data_in_train - mean_in) / std_in
data_out_train_scaled = (data_out_train - mean_out) / std_out
data_in_test_scaled = (data_in_test - mean_in) / std_in
data_out_test_scaled = (data_out_test - mean_out) / std_out



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
    

train_dataset = dataset(data_in_train_scaled, data_out_train_scaled)
test_dataset = dataset(data_in_test_scaled, data_out_test_scaled)


in_cols = data_in_train_scaled.shape[1]
out_cols = data_out_train_scaled.shape[1]

#################


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

        self.patch_size = 3
        self.num_patches =  in_cols//3
        self.num_outputs = out_cols
        self.embed_dim = model_args["embed_dim"]
        self.hidden_dim = model_args["hidden_dim"]
        self.num_heads = model_args["num_heads"]
        self.num_layers = model_args["num_layers"]
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
        #self.save_hyperparameters()
        self.learning_rate = model_args["learning_rate"]
        self.model = NetworkTransformer(model_args)

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
        self.log('train_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        return y_pred
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.custom_loss(y, y_pred)
        self.log('val_loss', loss)
        return y_pred


    def custom_loss(self, y_true, y_pred):
        err = y_pred - y_true
        err = (torch.abs(err)) + torch.pow(err, 2)
        return torch.mean(err)

class SMAC_model:
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges.
        # To illustrate different parameter types, we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace()

        max_layers = 8

        num_layers = Integer("num_layers", (2, max_layers), default=4)
        num_heads = Categorical("num_heads", [1, 2, 4, 8], default=4)

        embed_dim = Categorical("embed_dim", [32, 64, 128, 256, 512, 1024], default=256)
        hidden_dim = Categorical("hidden_dim", [64, 128, 256, 512, 1024], default=512)

        dropout_values=[0]
        for i in range(0,5):
            for j in range(1,2):
                dropout_values.append(10**(-j)*(i+1))
        dropout = Categorical("dropout", dropout_values, default=0.1)
        

        batch_size = Categorical("batch_size", [256, 512, ], default=512)
        # create list of possible learning rates
        learning_rate=[]
        for i in range(0,9):
            for j in range(2,6):
                learning_rate.append(10**(-j)*(i+1))
        learning_rate = Categorical("learning_rate", learning_rate, default=0.0001)

        # Add all hyperparameters at once:
        cs.add_hyperparameters([num_layers, num_heads, embed_dim, hidden_dim, batch_size, learning_rate, dropout])

        # We can also add multiple conditions on hyperparameters at once:
        #cs.add_conditions([ use_batch_size, use_lr_init])
        return cs
    
    def train(self, config: Configuration, seed: int = 42, budget: int = 50) -> float:
        # For deactivated parameters (by virtue of the conditions),
        # the configuration stores None-values.
        # This is not accepted by the MLP, so we replace them with placeholder values.
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0) # num_worker should be set to 0 if you use also num_workers>1 in smac
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = Network_Transformer(config)
            trainer = pl.Trainer(devices=1, max_epochs=int(np.ceil(budget)), deterministic=True, enable_progress_bar=False, enable_checkpointing=False, enable_model_summary=False,logger =False,)
            trainer.fit(model=model, train_dataloaders=train_loader)
            # predict on validation set
            predictions = trainer.predict(model=model, dataloaders=test_loader)

            predictions = np.vstack(predictions)

            #NOTE: mean and std from all data(train + val)

            # rescacle predictions
            y_pred = predictions * std_out + mean_out

            mae = np.mean(np.abs(y_pred - data_out_test), axis=0)
            
            #smae = mae / std_out
            smae = np.mean(np.abs(predictions - data_out_test_scaled), axis=0)
        return np.mean(smae)


def plot_trajectory(facades: list[AbstractFacade]) -> None:
    """Plots the trajectory (incumbents) of the optimization process."""
    plt.figure()
    plt.title("Trajectory")
    plt.xlabel("Wallclock time [s]")
    plt.ylabel(facades[0].scenario.objectives)
    #plt.ylim(0, 0.4)

    for facade in facades:
        X, Y = [], []
        for item in facade.intensifier.trajectory:
            # Single-objective optimization
            assert len(item.config_ids) == 1
            assert len(item.costs) == 1

            y = item.costs[0]
            x = item.walltime

            X.append(x)
            Y.append(y)

        plt.plot(X, Y, label=facade.intensifier.__class__.__name__)
        plt.scatter(X, Y, marker="x")

    plt.legend()
    # save plot
    plt.savefig("figures/trajectory_"+str(exp_name)+".png")



if __name__ == "__main__":
    model = SMAC_model()


    facades: list[AbstractFacade] = []
    # Define our environment variables
    scenario = Scenario(
        model.configspace,
        name=exp_name,
        walltime_limit=3600*24*7,  # After 60 seconds, we stop the hyperparameter optimization
        n_trials=args.num_trials,  # Evaluate max 500 different trials
        min_budget=5,  # Train the Net using a hyperparameter configuration for at least 5 epochs
        max_budget=args.max_budget,  # Train the Net using a hyperparameter configuration for at most 50 epochs
        n_workers=10, # Use all available compute resources: os.cpu_count()
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=16)

    # Create our intensifier
    intensifier = Hyperband(scenario, incumbent_selection="highest_budget")

    # Create our SMAC object and pass the scenario and the train method
    smac = MFFacade(
        scenario,
        model.train,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
    )

    # Let's optimize
    incumbent = smac.optimize()

    # Get cost of default configuration
    #default_cost = smac.validate(model.configspace.get_default_configuration())
    #print(f"Default cost ({intensifier.__class__.__name__}): {default_cost}")

    # Let's calculate the cost of the incumbent

    #incumbent_cost = smac.validate(incumbent)
    #print(f"Incumbent cost ({intensifier.__class__.__name__}): {incumbent_cost}")
    print("Done")
    print("Incumbent: ",incumbent)
    facades.append(smac)

    # Let's plot it
    plot_trajectory(facades)

    # kill python workers with pkill  #FIXME: got a timeoutError (see bellow) add it as issue in SMAC3 github
    """
    2023-12-20 00:22:34,687 - tornado.application - ERROR - Exception in callback functools.partial(<bound method IOLoop._discard_future_result of <tornado.platform.asyncio.AsyncIOMainLoop object at 0x7f348e8f5df0>>, <Task finished name='Task-719539' coro=<SpecCluster._correct_state_internal() done, defined at /home/wadhah.zai/anaconda3/envs/biotacPlugin/lib/python3.9/site-packages/distributed/deploy/spec.py:346> exception=TimeoutError()>)
    Traceback (most recent call last):
    File "/home/wadhah.zai/anaconda3/envs/biotacPlugin/lib/python3.9/asyncio/tasks.py", line 490, in wait_for
        return fut.result()
    asyncio.exceptions.CancelledError

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
    File "/home/wadhah.zai/.local/lib/python3.9/site-packages/tornado/ioloop.py", line 738, in _run_callback
        ret = callback()
    File "/home/wadhah.zai/.local/lib/python3.9/site-packages/tornado/ioloop.py", line 762, in _discard_future_result
        future.result()
    File "/home/wadhah.zai/anaconda3/envs/biotacPlugin/lib/python3.9/site-packages/distributed/deploy/spec.py", line 448, in _close
        await self._correct_state()
    File "/home/wadhah.zai/anaconda3/envs/biotacPlugin/lib/python3.9/site-packages/distributed/deploy/spec.py", line 359, in _correct_state_internal
        await asyncio.gather(*tasks)
    File "/home/wadhah.zai/anaconda3/envs/biotacPlugin/lib/python3.9/site-packages/distributed/nanny.py", line 619, in close
        await self.kill(timeout=timeout, reason=reason)
    File "/home/wadhah.zai/anaconda3/envs/biotacPlugin/lib/python3.9/site-packages/distributed/nanny.py", line 396, in kill
        await self.process.kill(reason=reason, timeout=0.8 * (deadline - time()))
    File "/home/wadhah.zai/anaconda3/envs/biotacPlugin/lib/python3.9/site-packages/distributed/nanny.py", line 874, in kill
        await process.join(max(0, deadline - time()))
    File "/home/wadhah.zai/anaconda3/envs/biotacPlugin/lib/python3.9/site-packages/distributed/process.py", line 330, in join
        await wait_for(asyncio.shield(self._exit_future), timeout)
    File "/home/wadhah.zai/anaconda3/envs/biotacPlugin/lib/python3.9/site-packages/distributed/utils.py", line 1910, in wait_for
        return await asyncio.wait_for(fut, timeout)
    File "/home/wadhah.zai/anaconda3/envs/biotacPlugin/lib/python3.9/asyncio/tasks.py", line 492, in wait_for
        raise exceptions.TimeoutError() from exc
    asyncio.exceptions.TimeoutError
    """

    os.system("pkill -fU wadhah.zai python3.9")

    # stop and exit
    sys.exit(0)

