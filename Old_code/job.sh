#!/bin/bash
IFS=

#source path/to/miniconda3/bin/activate tensorflow1.11.0-k80

echo $(python -V 2>&1)
echo $(nvcc --version 2>&1)

echo $(python /sps/km3net/users/ffilippi/ML/CNN_3d.py 2>&1)
