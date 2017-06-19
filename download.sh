#!/usr/bin/env bash

if ! [ -d data ]
then
    mkdir data
fi
    
cd data
../gdown.pl 'https://drive.google.com/uc?export=download&id=0BzqR-ZhARFEVcXpGLXNDaDdXc00' sample_submission.csv.zip
../gdown.pl 'https://drive.google.com/uc?export=download&id=0BzqR-ZhARFEVSklTeXc0WVZSMG8' train_labels.csv.zip
../gdown.pl 'https://drive.google.com/uc?export=download&id=0BzqR-ZhARFEVTWxjcS1DSHhzZjA' train.7z
../gdown.pl 'https://drive.google.com/uc?export=download&id=0BzqR-ZhARFEVMFVSZ0FXSjY4MVE' test.7z

unzip sample_submission.csv.zip
unzip train_labels.csv.zip
7z x train.7z
7z x test.7z
cd ..

