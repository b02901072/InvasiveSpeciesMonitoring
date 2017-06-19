#!/usr/bin/env bash

if ! [ -d data ]
then
    mkdir data
    cd data
    wget https://www.kaggle.com/c/invasive-species-monitoring/download/sample_submission.csv.zip
    wget https://www.kaggle.com/c/invasive-species-monitoring/download/test.7z
    wget https://www.kaggle.com/c/invasive-species-monitoring/download/train.7z
    wget https://www.kaggle.com/c/invasive-species-monitoring/download/train_labels.csv.zip

    unzip sample_submission.csv.zip
    unzip train_labels.csv.zip
    7z x test.7z
    7z x train.7z
fi

