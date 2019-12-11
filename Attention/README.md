This folder contains the code for attention + LSTM

In order to run it, use:
$ python main_attn.py --data_dir ./data/COCO/ --output_dir <xxx> --embedding_length 512 --optimizer_type Adam --lr 0.0001 --alpha_c 1.0 --num_epochs 100 --shuffle True --num_workers 16 --is_training 1 --is_testing 0
