#!/usr/bin/env bash
search_dir=/sps/km3net/users/ffilippi/ML/nu_gehen/nu_root

for entry in "$search_dir"/*
do 
echo "$entry"
#echo "$entry" |grep -o "[0-9]\+" 
var="${entry#*0000}"
#echo "$var"
number="${var%%.*}"
echo "$number"
vers="${var#*.}"
version="${vers%%.*}"
echo "$version"

VAR2="KM3NeT_00000042_0000"
VAR3="$VAR2$number.detx"
VAR4="nu_mu_$version.h5"
cp /sps/km3net/repo/mc/atm_neutrino/KM3NeT_00000042/v5.1/trigger/"$VAR3" .
tohdf5 -o /sps/km3net/users/ffilippi/ML/nu_gehen/"$VAR4" "$entry"
calibrate "$VAR3" /sps/km3net/users/ffilippi/ML/nu_gehen/"$VAR4"
rm -- *.detx
done


