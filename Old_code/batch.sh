#!/bin/bash
IFS=

#singularity exec --nv --bind /pbs:/pbs /cvmfs/singularity.in2p3.fr/images/HPC/GPU/centos7_cuda9-2_cudnn7-3_nccl2-2-12.simg /sps/km3net/users/ffilippi/ML/job.sh
singularity exec --nv --bind /pbs:/pbs --bind /sps:/sps  /cvmfs/singularity.in2p3.fr/images/HPC/GPU/centos7_cuda10-0_cudnn7-6-5_nccl2-5-6.sif /sps/km3net/users/ffilippi/ML/jobscript.sh
