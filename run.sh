#!/bin/bash

if [ "$1" == "train" ]; then
    CUDA_VISIBLE_DEVICES=0 python run.py train
elif [ "$1" == "eval" ]; then
    CUDA_VISIBLE_DEVICES=0 python run.py eval
elif [ "$1" == "test" ]; then
    CUDA_VISIBLE_DEVICES=0 python run.py test
elif [ "$1" == "train_debug" ]; then
    python run.py train
else
    echo "Invalid Option Selected"
fi
