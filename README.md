### Implementation in PyTorch

Done as part of the Course Project for IE-534/CS-547 (Deep Learning).

#### Contributors:
- Akhil Gupta
- Gregory Romanchek
- Heba Flemban
- Moitreya Chatterjee
- Yan Zhang
- Zihao Yang


### Dependencies (with version numbers):
- pycocotools      2.0.0    
- python           3.7.5
- python-dateutil  2.8.1
- pytorch          1.1.0
- qt               5.9.7
- readline         7.0
- scikit-learn     0.21.3
- scipy            1.3.1 
- six              1.13.0
- torchvision      0.3.0
- nltk

* Clone this repository 

### Dataset Pre-processing
First download the COCO [training](https://images.cocodataset.org/zips/train2014.zip) and [validation](https://images.cocodataset.org/zips/val2014.zip) image sets and the [annotation](images.cocodataset.org/annotations/annotations_trainval2014.zip) files as well. Extract and keep them in a folder called COCO. This folder is our data folder.

### Execution Instruction :
The code has three main components. We provide the instructions for running them here, the full choice of different arguments maybe obtained the argparse in the main files:
* The GRU based model:
  - Training Regime:
  $ python main.py --data_dir <path/to/dataset/root> --output_dir <path/to/output/directory> --embedding_length 512 --optimizer_type Adam --lr 0.0001 --num_epochs 100 --shuffle True --num_workers 16 --is_training 1 --is_testing 0
  - Testing Regime:
  $ python main.py --data_dir <path/to/dataset/root> --output_dir <path/to/output/directory> --load_model_test <model_name> --num_workers 16 --is_training 0 --is_testing 1
  
* The LSTM based model (inside the LSTM folder)
  - Training Regime:
  $ python main_lstm.py --data_dir <path/to/dataset/root> --output_dir <path/to/output/directory> --embedding_length 512 --optimizer_type Adam --lr 0.0001 --num_epochs 100 --shuffle True --num_workers 16 --is_training 1 --is_testing 0
  - Testing Regime:
  $ python main_lstm.py --data_dir <path/to/dataset/root> --output_dir <path/to/output/directory> --load_model_test <model_name> --num_workers 16 --is_training 0 --is_testing 1

* The Attention based model (inside the Attention folder)
  - Training Regime (GRU + Attention):
  $ python main_attn.py --data_dir <path/to/dataset/root> --output_dir <path/to/output/directory> --embedding_length 512 --optimizer_type Adam --lr 0.0001 --num_epochs 100 --alpha_c 1.0 --shuffle True --num_workers 16 --is_training 1 --is_testing 0
  - Testing Regime:
  $ python main_attn.py --data_dir <path/to/dataset/root> --output_dir <path/to/output/directory> --load_model_test <model_name> --num_workers 16 --is_training 0 --is_testing 1
  
  - Training Regime (LSTM + Attention):
  $ python main_attn_LSTM.py --data_dir <path/to/dataset/root> --output_dir <path/to/output/directory> --embedding_length 512 --optimizer_type Adam --lr 0.0001 --num_epochs 100 --alpha_c 1.0 --shuffle True --num_workers 16 --is_training 1 --is_testing 0
  - Testing Regime:
  $ python main_attn_LSTM.py --data_dir <path/to/dataset/root> --output_dir <path/to/output/directory> --load_model_test <model_name> --num_workers 16 --is_training 0 --is_testing 1
