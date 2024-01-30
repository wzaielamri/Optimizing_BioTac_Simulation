#!/bin/bash
#conda activate OptBioSim


python3.9 XGBoost_SMAC_pytorch.py  --window_before 10 --window_after 10 --specific True  --max_budget 30
python3.9 XGBoost_SMAC_pytorch.py  --window_before 10 --window_after 0  --specific True --max_budget 30
python3.9 XGBoost_SMAC_pytorch.py  --window_before 0 --window_after 0 --specific True --max_budget 30  
python3.9 XGBoost_SMAC_pytorch.py  --window_before 10 --window_after 10 --specific True --add_five True  --prefix _+5-5 --max_budget 30
python3.9 XGBoost_SMAC_pytorch.py  --window_before 10 --window_after 10  --max_budget 30
python3.9 XGBoost_SMAC_pytorch.py  --window_before 10 --window_after 0  --max_budget 30
python3.9 XGBoost_SMAC_pytorch.py  --window_before 10 --window_after 10  --last_pos True --next_pos True --specific True  --max_budget 30
python3.9 XGBoost_SMAC_pytorch.py  --window_before 10 --window_after 10 --last_pos True --next_pos True  --max_budget 30
python3.9 XGBoost_SMAC_pytorch.py  --window_before 10 --window_after 0 --last_pos True  --max_budget 30

####################################


python3.9 FFNN_SMAC_pytorch.py  --window_before 10 --window_after 10 --specific True --prefix _smac30epochs --max_budget 30
python3.9 FFNN_SMAC_pytorch.py  --window_before 10 --window_after 0  --specific True --prefix _smac30epochs --max_budget 30
python3.9 FFNN_SMAC_pytorch.py  --window_before 0 --window_after 0 --specific True --prefix _smac30epochs --max_budget 30  
python3.9 FFNN_SMAC_pytorch.py  --window_before 10 --window_after 10 --specific True --add_five True  --prefix _+5-5_smac30epochs --max_budget 30
python3.9 FFNN_SMAC_pytorch.py  --window_before 10 --window_after 10 --prefix _smac30epochs --max_budget 30
python3.9 FFNN_SMAC_pytorch.py  --window_before 10 --window_after 0 --prefix _smac30epochs --max_budget 30
python3.9 FFNN_SMAC_pytorch.py  --window_before 10 --window_after 10  --last_pos True --next_pos True --specific True --prefix _smac30epochs --max_budget 30
python3.9 FFNN_SMAC_pytorch.py  --window_before 10 --window_after 10 --last_pos True --next_pos True --prefix _smac30epochs --max_budget 30
python3.9 FFNN_SMAC_pytorch.py  --window_before 10 --window_after 0 --last_pos True --prefix _smac30epochs --max_budget 30


####################################

python3.9 Transformer_SMAC_pytorch.py  --window_before 10 --window_after 10 --specific True --prefix _smac30epochs --max_budget 30
python3.9 Transformer_SMAC_pytorch.py  --window_before 10 --window_after 0  --specific True --prefix _smac30epochs --max_budget 30
python3.9 Transformer_SMAC_pytorch.py  --window_before 0 --window_after 0 --specific True --prefix _smac30epochs --max_budget 30  
python3.9 Transformer_SMAC_pytorch.py  --window_before 10 --window_after 10 --specific True --add_five True  --prefix _+5-5_smac30epochs --max_budget 30
python3.9 Transformer_SMAC_pytorch.py  --window_before 10 --window_after 10 --prefix _smac30epochs --max_budget 30
python3.9 Transformer_SMAC_pytorch.py  --window_before 10 --window_after 0 --prefix _smac30epochs --max_budget 30
python3.9 Transformer_SMAC_pytorch.py  --window_before 10 --window_after 10  --last_pos True --next_pos True --specific True --prefix _smac30epochs --max_budget 30
python3.9 Transformer_SMAC_pytorch.py  --window_before 10 --window_after 10 --last_pos True --next_pos True --prefix _smac30epochs --max_budget 30
python3.9 Transformer_SMAC_pytorch.py  --window_before 10 --window_after 0 --last_pos True --prefix _smac30epochs --max_budget 30

