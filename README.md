This is the code and data for the BMVC'20 paper [Learning to Abstract and Predict Human Actions](https://www.bmvc2020-conference.com/conference/papers/paper_0979.html).

# Download Data
Please download the Hierarchical Breakfast from the link below and put the folder `HierarchicalBreakfast` folder 
inside a `data` directory in this current directory (i.e. `./data/HierarchicalBreakfast/...`).

Link: [Hierarchical Breakfast](https://bit.ly/3xu7Gy6).

For the original Breakfast Actions dataset annotation, and other related material (e.g. videos), please access 
[Breakfast Actions](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/).

# Environment Setup
First please create an appropriate environment using conda: 

> conda env create -f environment.yml

> conda activate fpua

# Test Pre-Trained Models
To evaluate pre-trained models run the `test.py` script.
A few examples:

#### Evaluate HERA
> CUDA_VISIBLE_DEVICES=0 python -W ignore test.py hera_cv --pretrained_root ./pretrained 
>--pretrained_suffix hera/hs-16-16_20e_bs512_act-elu-sigmoid_h2_lr1e-03_tlr1e-03_es-16-16_tanh_tsalways_mtll_use-hmgruv2_mv3_aips_wfa_isq35_osq50_nc5_atleast20_with-val.tar 
>--fine_labels_root_path ./data/HierarchicalBreakfast/labels/fine 
>--coarse_labels_root_path ./data/HierarchicalBreakfast/labels/coarse 
>--fine_action_to_id ./data/HierarchicalBreakfast/dictionaries/fine_action_to_id.txt 
>--coarse_action_to_id ./data/HierarchicalBreakfast/dictionaries/coarse_action_to_id.txt 
>--observed_fraction 0.2 --ignore_silence_action silence

#### Evaluate Dummy Predictor on the fine level of the Hierarchical Breakfast
> CUDA_VISIBLE_DEVICES=0 python -W ignore test.py dummy_cv --labels_root_path ./data/HierarchicalBreakfast/labels/fine 
>--action_to_id ./data/HierarchicalBreakfast/dictionaries/fine_action_to_id.txt 
>--observed_fraction 0.2 --unobserved_fraction 0.5 --ignore silence

#### Evaluate Independent-Single-RNN on coarse level of the Hierarchical Breakfast
> CUDA_VISIBLE_DEVICES=0 python -W ignore test.py baselines_cv --pretrained_root ./pretrained 
>--pretrained_suffix baselines/hs16_75e_bs12_act-sigmoid_h1_lr1e-03_es16_tanh_mtll_wfa_bt0coarse_tm_isq35_osq35_with-val.tar 
>--fine_labels_root_path ./data/HierarchicalBreakfast/labels/fine 
>--coarse_labels_root_path ./data/HierarchicalBreakfast/labels/coarse 
>--fine_action_to_id ./data/HierarchicalBreakfast/dictionaries/fine_action_to_id.txt 
>--coarse_action_to_id ./data/HierarchicalBreakfast/dictionaries/coarse_action_to_id.txt 
>--observed_fraction 0.2 --ignore_silence_action silence


# Train a Model
To train a model run the `train.py` script (use `-h` flag for options). See below an example on how to train the provided 
pre-trained `HERA` model (for full cross-validation you need to run the same command on the other training splits as well):

> CUDA_VISIBLE_DEVICES=0 python train.py hera --training_data ./data/HierarchicalBreakfast/training_arrays/hera/split02-03-04T_isq35_osq50_nc5_atleast20_nosilence_withfinalaction.npz 
>--validation_data ./data/HierarchicalBreakfast/training_arrays/hera/split02-03-04V_isq35_osq50_nc5_atleast20_nosilence_withfinalaction.npz 
>--epochs 20 --multi_task_loss_learner --always_include_parent_state --log_dir ./pretrained/split02-03-04/hera

The training arrays are provided here for convenience and were extracted from the `./data/HierarchicalBreakfast/labels` 
using the `process_data.py` script.
