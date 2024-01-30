#!/bin/bash
#conda activate OptBioSim



python netB_train_pytorch_CV_findBestTemp_testvaltrain.py --window_before 10 --window_after 10 --specific True  
python netB_train_pytorch_CV_findBestTemp_testvaltrain.py --window_before 10 --window_after 0  --specific True 
python netB_train_pytorch_CV_findBestTemp_testvaltrain.py --window_before 0 --window_after 0 --specific True  
python netB_train_pytorch_CV_findBestTemp_testvaltrain.py --window_before 10 --window_after 10 --specific True  --add_five True
python netB_train_pytorch_CV_findBestTemp_testvaltrain.py --window_before 10 --window_after 10 
python netB_train_pytorch_CV_findBestTemp_testvaltrain.py --window_before 10 --window_after 0 
python netB_train_pytorch_CV_findBestTemp_testvaltrain.py --window_before 10 --window_after 10  --last_pos True --next_pos True --specific True  
python netB_train_pytorch_CV_findBestTemp_testvaltrain.py --window_before 10 --window_after 10 --last_pos True --next_pos True


#####################################

python XGBoost_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10 --specific True  --load_smac True  
python XGBoost_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 0  --specific True --load_smac True  
python XGBoost_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 0 --window_after 0 --specific True  --load_smac True 
python XGBoost_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10 --specific True --add_five True  --load_smac True --prefix _+5-5 
python XGBoost_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10 --load_smac True  
python XGBoost_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 0 --load_smac True  
python XGBoost_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10  --last_pos True --next_pos True --specific True  --load_smac True  
python XGBoost_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10 --last_pos True --next_pos True --load_smac True  
python XGBoost_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 0 --last_pos True --load_smac True  


####################################


python FFNN_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10 --specific True  --load_smac True --prefix _smac30epochs 
python FFNN_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 0  --specific True --load_smac True  --prefix _smac30epochs 
python FFNN_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 0 --window_after 0 --specific True  --load_smac True --prefix _smac30epochs 
python FFNN_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10 --specific True --add_five True  --load_smac True --prefix _+5-5_smac30epochs 
python FFNN_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10 --load_smac True --prefix _smac30epochs 
python FFNN_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 0 --load_smac True --prefix _smac30epochs 
python FFNN_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10  --last_pos True --next_pos True --specific True  --load_smac True --prefix _smac30epochs 
python FFNN_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10 --last_pos True --next_pos True --load_smac True --prefix _smac30epochs 
python FFNN_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 0 --last_pos True --load_smac True --prefix _smac30epochs 


####################################

python Transformer_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10 --specific True  --load_smac True --prefix _smac30epochs 
python Transformer_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 0  --specific True --load_smac True  --prefix _smac30epochs 
python Transformer_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 0 --window_after 0 --specific True  --load_smac True --prefix _smac30epochs 
python Transformer_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10 --specific True --add_five True  --load_smac True --prefix _+5-5_smac30epochs 
python Transformer_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10 --load_smac True --prefix _smac30epochs 
python Transformer_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 0 --load_smac True --prefix _smac30epochs 
python Transformer_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10  --last_pos True --next_pos True --specific True  --load_smac True --prefix _smac30epochs 
python Transformer_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 10 --last_pos True --next_pos True --load_smac True --prefix _smac30epochs 
python Transformer_train_pytorch_CV_SMACBest_testvaltrain.py --window_before 10 --window_after 0 --last_pos True --load_smac True --prefix _smac30epochs 

