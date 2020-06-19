# Software and computing project
Use and development of libraries implemented by KM3NeT collaboration for Machine Learning. Build a CNN for discrimination between muons and background:

https://git.km3net.de/ml/OrcaSong

https://git.km3net.de/ml/OrcaNet

The KM3NeT/ARCA detector is composed, in its final configuration of 230 strings deployed at a depth of 3500 m under the sea level. In order to reduce the electronic components deployed at the botttom of the sea, all the data (composed mainly by environmental background) are sent on-shore and then processed from high level trigger algorithms (L1 and L2). Between these algorithms there is a muon reconstruction procedure based on a pre-fit of the photon hits impinging on a single detector string. The informations produced on different strings are then grouped together in an offline fit. A fast online algorithm is crucial for a neutrino telescope in order to produce allerts sent then to other observatories for the multimessanger campaign. 

The main feature of this work is the training of a CNN with a single string detector MonteCarlo files and then compare its performances w.r.t. the usual algorithms applied for the muon reconstruction procedure.
The steps required for the pre-processing of the data are:

``` tohdf5 -o file.h5 /pathToMCFiles @ CC-IN2P3/
calibrate file.h5 /pathToDetectorFile/detectorfile.detx```

