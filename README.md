# Optimizing BioTac Simulation for Realistic Tactile Perception


This repository contains the code and the appendix for the paper [Optimizing BioTac Simulation for Realistic Tactile Perception](https://to.do) by Wadhah Zai El Amri and Nicolás Navarro-Guerrero.


# Abstract
Tactile sensing presents a promising opportunity for enhancing the interaction capabilities of today’s robots. BioTac is a commonly used tactile sensor that enables robots to perceive and respond to physical tactile stimuli. However, the sensor’s non-linearity poses challenges in simulating its behavior. In this paper, we first investigate a BioTac simulation that uses temperature, force, and contact point positions to predict the sensor outputs. We show that training with BioTac temperature readings does not yield accurate sensor output predictions during deployment. Consequently, we tested three alternative models, i.e., an XGBoost regressor, a neural network, and a transformer encoder. We train these models without temperature readings and provide a detailed investigation of the window size of the input vectors. We demonstrate that we achieve statistically significant improvements over the baseline network. Furthermore, our results reveal that the XGBoost regressor and transformer outperform traditional feed-forward neural networks in this task. We make all our code and results available online on [GitHub](https://github.com/wzaielamri/Optimizing_BioTac_Simulation/).

## Appendix
The appendix is available [here](https://github.com/wzaielamri/Optimizing_BioTac_Simulation/blob/master/Appendix_Simulating_Tactile_Signals_for_the_SynTouch_BioTac_Sensor_Paper.pdf).

# Install

Install the conda package manager from https://docs.conda.io/en/latest/miniconda.html

```
# Required: 
conda create -n OptBioSim python=3.9
conda activate OptBioSim
git clone https://github.com/wzaielamri/Optimizing_BioTac_Simulation
cd Optimizing_BioTac_Simulation
pip install -r requirements.txt
```

# Usage

- Activate the environment:

```
conda activate OptBioSim
```

- To search for the best hyperparameters, using SMAC, run the following command:

```
./script_search_smac.sh
```

- To train the best model, run the following command:

```
./script_train.sh
```

- To test the best model (inference and FLOS), run the following command:

```
./script_calculate_inference.sh
./script_calculate_flops.sh
```

- To plot the results, use the notebooks provided in the folder: *generate_figures*.

## Dataset

The dataset used is available [here](https://tams.informatik.uni-hamburg.de/research/datasets/index.php#biotac_single_contact_response) (Ruppel et al. 2018).

## Checkpoints and Results

The checkpoints and results are available upon request.


# Copyrights:

Accepted at the International Joint Conference on Neural Network (IJCNN) 2024, Yokohama, Japan.
Note: ©2024 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.


# Citation

```
@inproceedings{ZaiElAmri2024Optimizing,
  title = {Optimizing {{BioTac Simulation}} for {{Realistic Tactile Perception}}},
  booktitle = {International {{Joint Conference}} on {{Neural Networks}} ({{IJCNN}})},
  author = {Zai El Amri, Wadhah and {Navarro-Guerrero}, Nicol{\'a}s},
  year = {2024},
  publisher = {IEEE},
  address = {Yokohama, Japan},
  annotation = {https://github.com/wzaielamri/Optimizing\_BioTac\_Simulation}
}
```
