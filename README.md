
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/a6a24537656b4fb5abc9bfe37a355e19)](https://app.codacy.com/manual/francescofilippini6/Software_computer?utm_source=github.com&utm_medium=referral&utm_content=francescofilippini6/Software_computer&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://travis-ci.org/francescofilippini6/Software_computer.svg?branch=master)](https://travis-ci.org/francescofilippini6/Software_computer)
[![Coverage Status](https://coveralls.io/repos/github/francescofilippini6/Software_computer/badge.svg?branch=master)](https://coveralls.io/github/francescofilippini6/Software_computer?branch=master)
# Software and computing project 

Use and development of libraries implemented by KM3NeT collaboration for Machine Learning. Build of a CNN for discrimination between muons and neutrinos. Further informations:

<https://git.km3net.de/ml/OrcaSong>

<https://git.km3net.de/ml/OrcaNet>

The KM3NeT ARCA and ORCA detectors are composed, in their final configuration of 230 and 115 strings respectively deployed at a depth of 3500 m under the sea level. In order to reduce the electronic components deployed at the botttom of the sea, all the data (composed mainly by environmental background) are sent on-shore and then processed from high level trigger algorithms (L1 and L2). Between these algorithms there is a muon reconstruction procedure based on a pre-fit of the photon hits impinging on a single detector string. The informations produced on different strings are then grouped together in an offline fit. A fast online algorithm is crucial for a neutrino telescope in order to produce allerts sent then to other observatories for the multimessanger campaign. 

The main feature of this work is the training of a CNN with a single string detector MonteCarlo files and then compare its performances w.r.t. the usual algorithms applied for the muon reconstruction procedure. The progressive construction of these detectors will allows to keep also on the "algorithm-side" the modularity required in the construction phase. Another aspect to build this type of network is the possibility of apply it on the data originated at the BCI (Bologna Common Infrastructure) consisting in a unique 1 string test-bench able to implement in all the aspects the features and characteristic of the detector itself. 
The steps necessary for the pre-processing of the data are:


then the calibration of the MonteCarlo files with the right detector file:

```calibrate file.h5 /pathToDetectorFile/detectorfile.detx```

At this point is possible to built the image of the event combining togheter the positional informations of the hit with its arrival time. Particular attention must be paid in the right choice of the binning intervals.
After the image creation we can start the training and test procedure for the construction of the CNN itself, based on Keras models.



Possibility of evaluate also a new type of network, Graph Neaural Network, based on the ParticleNet architecture.This possibility was introduced recently by the work of other KM3NeT members and there is still an open window to bring that framework from KM3NeT/ORCA also to KM3NeT/ARCA. 

Further documentation can be found at:
<https://arxiv.org/abs/1902.08570>
<https://arxiv.org/abs/2004.08254>
