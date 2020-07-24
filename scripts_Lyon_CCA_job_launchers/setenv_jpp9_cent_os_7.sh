#!/usr/bin/env bash

workdir=${PWD}

#Load ROOT v5
#ROOT_VERSION=5.34.23
#ROOT_VERSION=5.34.38
#source /usr/local/root/${ROOT_VERSION}/bin/thisroot.sh

# Load JPP
cd /pbs/throng/km3net/src/Jpp/v9.0.8454/
source /pbs/throng/km3net/src/Jpp/v9.0.8454/setenv.sh
cd "${workdir}"
