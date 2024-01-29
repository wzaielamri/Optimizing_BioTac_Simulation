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
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Integer,
)
from smac import MultiFidelityFacade as MFFacade
from smac import Scenario, HyperparameterOptimizationFacade
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from deepcave import Recorder, Objective

from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", type=str, default="", help="exp name")
parser.add_argument('--window_before', type=int, default=10, help='window before')
parser.add_argument('--window_after', type=int, default=10, help='window after')
parser.add_argument('--specific', type=bool, default=False, help='specific')
parser.add_argument('--next_pos', type=bool, default=False, help='next_pos')
parser.add_argument('--last_pos', type=bool, default=False, help='last_pos')
parser.add_argument('--num_trials', type=int, default=1000, help='number of smac trials to run')
parser.add_argument('--max_budget', type=int, default=30, help='max_budget')
parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
parser.add_argument('--add_five', type=bool, default=False, help='add_five')
parser.add_argument("--file", type=str, default="./biotac_single_contact_response/2018-01-19-18-16-58_biotac_ff_stick_calibration.bag.csv", help="file path")


args = parser.parse_args()

exp_name="SMACSearch_XGBoost_pytorch"
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

in_cols = data_in_train_scaled.shape[1]
out_cols = data_out_train_scaled.shape[1]

#################




class SMAC_model:
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges.
        # To illustrate different parameter types, we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace()

        eta               = UniformFloatHyperparameter( "eta", 0.0001, 0.5, default_value = 0.3)
        n_estimators      = UniformIntegerHyperparameter( "n_estimators", 100, 1000, default_value = 100)
        gamma             = UniformIntegerHyperparameter( "gamma", 0, 10, default_value = 0)
        max_depth         = UniformIntegerHyperparameter( "max_depth", 1, 10, default_value = 6)
        min_child_weight  = UniformIntegerHyperparameter( "min_child_weight", 1, 100, default_value = 1)
        max_delta_step    = UniformIntegerHyperparameter( "max_delta_step", 0, 10, default_value = 0)
        subsample         = UniformFloatHyperparameter( "subsample", 0.5, 1, default_value = 1)
        colsample_bytree  = UniformFloatHyperparameter( "colsample_bytree", 0.5, 1, default_value = 1)
        colsample_bylevel = UniformFloatHyperparameter( "colsample_bylevel", 0.5, 1, default_value = 1)
        colsample_bynode  = UniformFloatHyperparameter( "colsample_bynode", 0.5, 1, default_value = 1)

        cs.add_hyperparameters([eta, n_estimators, gamma, max_depth, min_child_weight, max_delta_step, subsample, colsample_bytree, colsample_bylevel, colsample_bynode])
        return cs

    def train(self, config: Configuration, seed: int = 42) -> float:
        # shuffle training data
        global data_in_train, data_out_train, data_in_train_scaled, data_out_train_scaled, data_in_test_scaled, data_out_test_scaled, mean_out, std_out
        idx = np.arange(data_in_train.shape[0])
        np.random.shuffle(idx)
        data_in_train = data_in_train[idx]
        data_in_train_scaled = data_in_train_scaled[idx]
        data_out_train = data_out_train[idx]
        data_out_train_scaled = data_out_train_scaled[idx]


        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            config_gpu = config.copy()
            config_gpu["config"] = "cuda"
            multi_reg = MultiOutputRegressor(
            estimator = XGBRegressor(objective="reg:absoluteerror",**config_gpu),
            #n_jobs = 1,
            )    
            multi_reg.fit(data_in_train_scaled, data_out_train_scaled)
            predictions = multi_reg.predict(data_in_test_scaled)

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
        n_workers=args.num_workers, # Use all available compute resources: os.cpu_count()
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=16)

    # Create our intensifier
    intensifier = Hyperband(scenario)

    # Create our SMAC object and pass the scenario and the train method
    smac = HyperparameterOptimizationFacade(
        scenario,
        model.train,
        initial_design=initial_design,
        #intensifier=intensifier,
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


