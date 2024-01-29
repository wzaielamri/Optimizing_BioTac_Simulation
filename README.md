# Optimizing BioTac Simulation for Realistic Tactile Perception


This repository contains the code and the appendix for the paper [Optimizing BioTac Simulation for Realistic Tactile Perception](https://to.do) by Wadhah Zai El Amri and Nicolás Navarro-Guerrero.



# Abstract
Tactile sensing presents a promising opportunity for enhancing the interaction capabilities of today’s robots. BioTac is a commonly used tactile sensor that enables robots to perceive and respond to physical tactile stimuli. However, the non-linearity of this sensor poses challenges in simulating its behavior. In this paper, we first investigate a BioTac simulation that uses temperature, force, and contact point positions to predict the sensor outputs. We show that training with BioTac temperature readings does not yield accurate sensor output predictions during deployment. Consequently, we tested three alternative models, i.e., an XGBoost regressor, a neural network, and a transformer encoder. We train these models without temperature readings and provide a detailed investigation of the window size of the input vectors. We demonstrate that we achieve an improvement of 7.5% over the baseline network. Furthermore, our results reveal that XGBoost regressor and transformer outperform traditional feed-forward neural networks in this task. We make all our code and results available online on [GitHub](https://github.com/wzaielamri/Optimizing_BioTac_Simulation/).


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

# Citation

```
@article{wzaielamri2024OptBioSim,
  title={Optimizing BioTac Simulation for Realistic Tactile Perception.},
  author={Wadhah Zai El Amri and Nicolás Navarro-Guerrero},
  journal={},
  year={2024}
}
```
