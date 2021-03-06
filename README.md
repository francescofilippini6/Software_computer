# Software and computing project 
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/a6a24537656b4fb5abc9bfe37a355e19)](https://app.codacy.com/manual/francescofilippini6/Software_computer?utm_source=github.com&utm_medium=referral&utm_content=francescofilippini6/Software_computer&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://travis-ci.org/francescofilippini6/Software_computer.svg?branch=master)](https://travis-ci.org/francescofilippini6/Software_computer)
[![Coverage Status](https://coveralls.io/repos/github/francescofilippini6/Software_computer/badge.svg?branch=master)](https://coveralls.io/github/francescofilippini6/Software_computer?branch=master)

Use and development of libraries implemented by KM3NeT collaboration for Machine Learning. Build of a CNN for discrimination between atmospheric muons (background) and neutrinos and for neutrino zenith reconstruction. Further informations:

<https://git.km3net.de/ml/OrcaSong>

<https://git.km3net.de/ml/OrcaNet>

The KM3NeT ARCA and ORCA detectors are composed, in their final configuration of 230 and 115 strings respectively deployed at a depth of 3500 m under the sea level. In order to reduce the electronic components deployed at the botttom of the sea, all the data are sent on-shore and then processed from high level trigger algorithms (L1 and L2). Between these algorithms there is a muon reconstruction procedure based on a pre-fit of the photon hits impinging on a single detector string. The informations produced on different strings are then grouped together in an offline fit. A fast online algorithm is crucial for a neutrino telescope in order to produce allerts sent then to other observatories for the multimessanger campaign. 
The main feature of this work is the training of a CNN with a single string detector MonteCarlo files in order to compare its performances w.r.t. the usual algorithms applied for the muon reconstruction procedure. The progressive construction of KM3NeT detectors will allows to keep also on the "algorithm-side" the modularity required in the construction phase. Another aspect to build this type of network is the possibility to apply it on the data originated at the BCI (Bologna Common Infrastructure) consisting in a unique 1 string test-bench able to implement in all the aspects the features and characteristics of the detector itself. 
## Data inspection
Considering the angular distribution of the events, neutrinos come from all the directions of the sky, traversing also all the Earth, and interacting at the rock at the base of the detector. Muons, at the contrary, are not able to traverse large amount of rock and therefore in KM3NeT these particles produce events coming only from the hemisphere over the detector itself (superposition of muon and neutrino events coming from above).
Previous works by the collaboration tried to make the identification without applying any zenithy cuts, but based on data simulated for a more complex detector configuration (4 strings and 115 strings configuration). In this work I follow the same workflow (no zenithy selection), but inside simulated data in single string configuration we have lots of symmetries on the signature of the events that cannot be disentagled due to the peculiar detector configuration. For this reason the final accuracy of the network on both the training and validation data, stucks at around 50-55 %, even changing hyperparameters (learning rate, optimizers, varying the architecture...)further info at the end.
For what concern the regression part on the zenith angle for neutrino direction, some results are shown for the one string configuration.
## Preliminary steps - Quick start
Git doesn't allow to upload files of large dimensions (even the extension LFS).The files needed are around 2GB and 3GB. Those here in the git folder are dummy ones created for testing and debug pourposes.

```clone the directory in a local folder```

and then run the script (after giving it the permission)

```./downloader.sh```

This will automatically download the complete files from google drive and set it correctly in the right folders. !!Attention!! In order to automatize the procedure the antivirus scan imposed by google drive for large file is bypassed. If you want you can download the file copying and paste the link: <https://docs.google.com/uc?export=download&confirm=Z5Zn&id=FILEID>, where FILEID is set inside inside the script downloader.sh.
## Data Preprocessing
All the MC files used for this work are stored @ Lyon CC.
The pre-processing of the data takes the majority of the time: 
inside **/nu_gehen** and **/mupage_root_files_from_irods** folders, there are the scripts used to copy locally the data, to transform it from .root in .h5 format (reducing the size to around 1/3 of the original) and then calibrated with the specific file. 
For muons were processed 1030 files each containing around 1500-1800 MC events.
For neutrinos were processed 1135 files each containing around 2000 MC events.
To inspect the .h5 file:

```ptdump filename.h5```

At this step, in order to further inspect physical parameter distribution of the MC events:

```tools/readMCinfo.py```

This will show the statistic of the events contained in the file (energy distribution, zenith distribution, position of the hit, direction of the track):
<img width="754" alt="Schermata 2020-07-26 alle 20 01 12" src="https://user-images.githubusercontent.com/58489839/88486016-d2ee1700-cf7a-11ea-8d74-5d96b78acbd2.png">
### Image creation
At this point we are ready to produce the images, and to bin the data thanks to:

```python binner.py 0 or 1```

0.  for muons processig
1.  for neutrinos processing

We list the previous directories in which are contained our files for neutrinos and muons respectively and all the files are passed to the Orcasong method ```FileBinner``` that produce new files, marked with "_hist_", stored respectively in **/outputfolder_mupage** and **/outputfolder_neutrino**. This method returns also the complete statistic of the events processed and binned, like below:
<img width="978" alt="Schermata 2020-07-26 alle 11 23 09" src="https://user-images.githubusercontent.com/58489839/88475681-7cf58100-cf32-11ea-9c4c-3259daac61d0.png">

For each event I select the only available informations:

1.  channel_id (PMT number in a Digital Optical Moduel) giving me some x-y direction information concerning the photon hits (there are 31 PMTs for each DOM);
2.  the z position of the photon hits (as done by previous works we have a DOM for each single bin along z = 18 bins);
3.  arrival time of the photons on the PMTs (the bin size = time resolution = 5ns grater than the time resolution achieved experimentally around 1 ns, but chosen to keep small the number of bins).

The muon pre-fit for a single line is based only on z-time information.
All the files produced in the previous step are then group together in two single files: **/outputfolder_mupage/concatenated.h5** and **/outputfolder_neutrino/concatenated.h5**, containing respectively 2x10^6 and 2.5x10^6 images.  
Always in the **/tools** folder the ```plottingImage.py ``` python program is able to create a z-t image of each event, to keep in touch with the workflow started, as below:
<img width="650" alt="Schermata 2020-07-26 alle 12 14 14" src="https://user-images.githubusercontent.com/58489839/88476564-93eba180-cf39-11ea-8d5a-3a31097545c2.png">

The final preprocessing step consists into the creation and appendding of a new dataset, called 'y', that represents the label (0 for muon data and 1 for neutrinos):
```python modify_h5_file_adding_label.py 0 or 1``` 
## CNN development
Due to the large files created (around 2 GB for the muon file and around 3GB for the neutrino one), I soon collide with the problem of fit the final dataset, and all the NN parameters in the RAM of the pc. I built therefore a custom data generator ```batch_uploader_keras.py``` , on the template given @ <https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>, that upload in the RAM only a batch of data ( = batch_size) per cycle. Making an observation of the memory usage at Lyon GPU cluster and also at googlecolab, thanks to memory_profiler I in fact observe the folowing:
<img width="716" alt="Schermata 2020-07-26 alle 12 26 07" src="https://user-images.githubusercontent.com/58489839/88476755-453f0700-cf3b-11ea-8439-ee440815b520.png">

@ **/tools/memory_profiling.txt** there are the instructions for making the memory profiling (Uncomment all the @profile decorator, to see the single function memory usage).

### Final results neutrino/muon discrimination
For sake of completeness, a previous trial was done with a smaller dataset O(10^5) images, taking into account only the z and time informations for the events. I reported below the architecture used and the accuracy - loss plots produced by ```tools/plot_history.py```. As we can see, even inscreasing the epoch number the network do not converge, reaching however on the training dataset a quite interesting accuracy.
<img width="804" alt="Schermata 2020-07-26 alle 12 42 57" src="https://user-images.githubusercontent.com/58489839/88477100-9a7c1800-cf3d-11ea-9140-4886f4242f12.png">

For this reason I preprocessed again the data, increasing the number of events and including also the information concerning the channel_id.
All the trainig steps were done @ google colab, on GPUs. The architecture is getting inspired from VGG models and by previous works done by the collaboration <https://inspirehep.net/literature/1791707>. The produced "costant" output is the one shown below, regardless the changes done in the learning rate, in the optimizer used and on the increased number of convolutional layers:
<img width="1302" alt="Schermata 2020-07-21 alle 19 13 56" src="https://user-images.githubusercontent.com/58489839/88477184-25f5a900-cf3e-11ea-9c68-32056316dce7.png">
As we can see the netwrok converge to an accuracy percentage around 50% (not at all interesting). Strong evindence seems to point out that the cause of this behaviour are symmetries inside the data, as already anticipated above.

## Regression problem on neutrino direction
After the failures in trying to discriminate between neutrinos and muons, I re-processed all the data, taking only z-t images. Thanks to the python program ```zenith_regression/cos_zenith_label.py```, I was able, for each binned file to attach a label, representing the cosine of the zenith angle of the neutrino direction (thanks to the km3pipe framework developed by the collaboration). 
At this point with a simple CNN architecture plus some hidden layers, in ```zenith_regression/regression_neutrino.py``` I obtain some interesting results (the activation function of the last layer is a softsign due to restrict the range in (-1,1) as the cosine itself):
<img width="401" alt="Schermata 2020-09-02 alle 17 52 07" src="https://user-images.githubusercontent.com/58489839/92300935-6db91880-ef5f-11ea-9c8f-f947c0289c8d.png">

Saving the weights in correspondence of the min validation loss, and reused it in the model we can predict some values and compare it to the MonteCarlo distribution:
<img width="951" alt="Schermata 2020-09-02 alle 21 54 25" src="https://user-images.githubusercontent.com/58489839/92300912-29c61380-ef5f-11ea-98d0-76dc50d8cb44.png">


Putting the point predicted and the real one in a scatter plot we obtain (shown only 500 points):
<img width="432" alt="Schermata 2020-09-02 alle 21 55 14" src="https://user-images.githubusercontent.com/58489839/92300920-45c9b500-ef5f-11ea-9d48-ab3fa66061c3.png">
and with a 2D representation all the predicted ones:
<img width="441" alt="Schermata 2020-09-02 alle 21 54 08" src="https://user-images.githubusercontent.com/58489839/92300927-5b3edf00-ef5f-11ea-990f-3092521d7682.png">

## Test routine
A simple test routine was implemented, in order to test the programs in the root directory. The process is keep automatized thanks to the usage of TRAVIS CI through the configuration file ```.travis.yml```, and thanks also to coveralls package for the coverage report. Also codacy was used in order to keep track and improve code quality (see the badge at the beginning).
## Future improvements
Under evaluation the possibility to apply a cut on the cosine(zenith), moving therefore from a muon-neutrino classification to an up/down classification. Also the possibility to simulate the random noise generated from the surrounding environment and add also this to the classification procedure (start to work on this but some problems found in the simulation procedure with the 1 string configuration detector).
