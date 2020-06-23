#!/usr/bin/env bash
VAR0=809
for ((X=8444; X<8495; X++ ))
do

VAR="mcv5.1.genhen_numuCC_10GeV.km3new_AAv1.jterbr0000"
VAR1="$VAR$X.$VAR0.root"
VAR2="KM3NeT_00000042_0000"
VAR3="$VAR2$X.detx"
VAR4="nu_mu$X.h5"
cp /sps/km3net/repo/mc/atm_neutrino/KM3NeT_00000042/v5.1/trigger/$VAR1 .
cp /sps/km3net/repo/mc/atm_neutrino/KM3NeT_00000042/v5.1/trigger/$VAR3 .
tohdf5 -o $VAR4 $VAR1
calibrate $VAR3 $VAR4
rm *.root
rm *.detx
((VAR0+=1))
done
