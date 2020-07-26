#!/bin/bash
cd outputfolder_mupage
rm concatenated.h5

fileid_muon="1fTd_EeBTuzjqiVJsSt2cBNvKRvuDlw6k"
filename_muon="concatenated.h5"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid_muon}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid_muon}" -o ${filename_muon}
rm cookie
cd ..

cd outputfolder_neutrino
rm concatenated.h5

fileid_neutrino="1V61tD0KuL4D4uyWeNyb5972C9fHpjfU4"
filename_neutrino="concatenated.h5"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid_neutrino}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid_neutrino}" -o ${filename_neutrino}
rm cookie
cd ..
