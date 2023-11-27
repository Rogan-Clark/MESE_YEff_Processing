## Scripts for YEff processing on L5 data

Quick and easy guide

1: Run RandomForestHDFMaker.py

    IN - .i3 files processed to L5

    OUT - .i3 and .hdf5 files with YEff info


    Preferably point this to /data/ana, as it makes an i3 file for each on you put in

    Configured to work on Snowstorm, MuonGun, and data


    RandomForest is, ironically, disabled for now


2: Run I3ToHdfCombiner.py or I3ToHdfCombiner_onepercentveto.py

    IN - .i3 files with YEff

    OUT - a single compiled hdf5 file


    Difference between the two is that the One percent veto calculates an alternative start for the YEff, based on

    a cascade starting at 1% of total reco deposited energy and not the first millipede cascade above 1 GeV



Everything else

FindHadronicCascadeExtent does what it says on the tin - finds the average and maximum positions of 
all particles produced from the hadronic cascade in CC interactions



Grid Files contains the dag and submission scripts to run on the IceCube cluster. Make sure they point to your directories



MESE utils contains bits and pieces necessary to run the MESE scripts. If running on the grid, you want these in an accessible shared cvmfs space

