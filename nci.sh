#!/bin/bash

baseLr=0.001 # initial learning rate
weight_decay=0.0001
nhid=128 # hidden dimension of features
dropout_ratio=0
edge_ratio=1.0 # the ratio to keep edges
pooling_ratio=0.5 # the ratio to keep nodes


python src/main.py --phase train --gpu_ids '0' --dataset 'NCI1' --modelname 'Co-Pooling' --batch_size 128 --test_batch 128 --save_flag False --pooling_ratio $pooling_ratio --baseLr $baseLr --weight_decay $weight_decay --nhid $nhid --edge_ratio $edge_ratio --eps 0 --dropout_ratio $dropout_ratio --alpha 0.1  --datapath './data/'

