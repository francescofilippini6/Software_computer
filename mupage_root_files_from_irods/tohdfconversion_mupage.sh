#!/usr/bin/env bash
search_dir=/sps/km3net/users/ffilippi/ML/mupage_root_files_from_irods/trigger

for entry in "$search_dir"/*
do 
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
VAR4="mupage_$version.h5"

cp /sps/km3net/repo/mc/atm_muon/KM3NeT_00000042/v5.1/trigger/"$VAR3" .
tohdf5 -o "/sps/km3net/users/ffilippi/ML/mupage_root_files_from_irods/$VAR4" "$entry"
calibrate "$VAR3" "/sps/km3net/users/ffilippi/ML/mupage_root_files_from_irods/$VAR4" 
rm *.detx
done


