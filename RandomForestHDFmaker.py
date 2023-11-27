#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT: combo/V00-00-03

## Makes the hdf files for NNMFit processing
## Stores all essential keys, including the reco'd Monopod/Millipede Track energies
## True data for simulated datasets
## And even the y_eff down below

from os import uname
from os import path as path
from os import system as system
from datetime import datetime
import sys,argparse,glob,time,os,gzip,pickle,urllib
print (uname())
import math
#from icecube import simclasses
from icecube import dataclasses,icetray,dataio,phys_services,VHESelfVeto,StartingTrackVeto, hdfwriter, recclasses, simclasses
from icecube.dataclasses import I3Particle
from icecube import MuonGun, millipede
from I3Tray import *
from numpy import sign as sign
#from numba import jit
import numpy as np
import time
import collections

os.environ["PATH"] += os.pathsep + "/cvmfs/icecube.opensciencegrid.org/users/msilva/python_packages/sklearn"
sys.path.append('/cvmfs/icecube.opensciencegrid.org/users/rclark/')
#sys.path.append("/home/rclark/scripts/RF_and_NNMScripts/")
sys.path.append('/cvmfs/icecube.opensciencegrid.org/users/msilva/python_packages')

from sklearn.externals import joblib

#sys.path.append("/cvmfs/icecube.opensciencegrid.org/users/msilva/python_packages/xgboost")
#sys.path.append("/cvmfs/icecube.opensciencegrid.org/users/msilva/python_packages/sklearn")
#sys.path.append("/cvmfs/icecube.opensciencegrid.org/users/msilva/python_packages/numpy")
#sys.path.append("/cvmfs/icecube.opensciencegrid.org/users/msilva/python_packages/scipy")

start_time = time.asctime()
print ('Started:', start_time)

### Pass initial arguments
desc="icetray script to extract info"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("-i","--infolder",   dest="infolder",    type=str,  default=[], nargs="+",help="[I]nfiles with frames")
parser.add_argument('-o','--outfilebase',dest='outfilebase', type=str,  default="", help='base name for [o]utfiles')
parser.add_argument("-k","--keep_gcd",   dest="keep_gcd",    type=int,  default=0, help="0: Don't save GCD with file 1: Do save GCD with file")
parser.add_argument("-u", '--use_grid',  dest="use_grid",  default=False, help='Use the gridftsp setup',  action='store_true')
parser.add_argument("-g", "--geofile",   dest="gcd",         type=str,  default="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz", help="GCD file")

args = parser.parse_args()
outfilebase=args.outfilebase
keep_gcd=args.keep_gcd


mass_dict=    {'2212': 1, #PPlus
               '1000020040': 4,#Alpha
               '1000070140': 14,#N14
               '1000130270': 27,#Al27
               '1000260560': 56#Fe56
              }

cascade_types = [I3Particle.Neutron,
                 I3Particle.Hadrons,
                 I3Particle.Pi0,
                 I3Particle.PiPlus,
                 I3Particle.PiMinus,
                 I3Particle.K0_Long,
                 I3Particle.KPlus,
                 I3Particle.KMinus,
                 I3Particle.PPlus,
                 I3Particle.PMinus,
                 I3Particle.K0_Short,
                 I3Particle.EMinus,
                 I3Particle.EPlus,
                 I3Particle.Gamma,
                 I3Particle.Brems,
                 I3Particle.DeltaE,
                 I3Particle.PairProd,
                 I3Particle.NuclInt]

import ESTES_general_utilities as Egu
infiles_temp = args.infolder

#my_grid, infiles = Egu.setup_fnames(args.use_grid, outfilebase, infiles_temp)

tray = I3Tray()

if args.use_grid == True:
    #print("Using grid")
    infiles = []
    my_grid = "gsiftp://gridftp.icecube.wisc.edu"
    #print(infiles_temp)
    if args.gcd == "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz":
        gcdFile = args.gcd
    else:
        gcdFile=my_grid+args.gcd
    infiles.append(gcdFile)
    for file in infiles_temp:
        infiles.append(my_grid + file)
    print(infiles)

else:
    print("Not using grid")
    infiles = infiles_temp
    gcdFile = args.gcd


###Define a surface boundary check
#surface_det = MuonGun.ExtrudedPolygon.from_file(gcdFile, padding=50)
import matplotlib.path as mpltPath
#bound_2D=[]
#x=[(surface_det.x[i],surface_det.y[i]) for i in range(len(surface_det.x))]
#bound_2D=mpltPath.Path(x)#Projection of detector on x,y plane
#def boundary_check(particle):
#    inlimit = False
#    if ((particle.pos.z <=max(surface_det.z)) and (particle.pos.z>=min(surface_det.z))):
#        if bound_2D.contains_points([(particle.pos.x, particle.pos.y)]):
#            inlimit=True
#    return inlimit

def select_MESE(geometry):
        r"""Select IceCube DOMs.
        Select all DOMs with an OM type `IceCube` from the given
        detector geometry and sort the selected DOMs per string in
        ascending depth.
        Parameters
        ----------
        geometry : I3OMGeoMap or dict(OMKey, tuple(I3OMGeo, ...))
            Detector geometry
        Returns
        -------
        dict(int, list(tuple(OMKey, I3OMGeo)))
            Mapping of string numbers to sequences of IceCube DOMs
            arranged in ascending depth.
        """
        strings = collections.defaultdict(list)
        # print (type(geometry))
        for omkey, omgeo in geometry.items():
            if np.iterable(omgeo):
                omgeo = omgeo[0]

            if omgeo.omtype == dataclasses.I3OMGeo.IceCube:
                strings[omkey.string].append((omkey, omgeo))

        for doms in strings.values():
            doms.sort(
                key=lambda omgeo: omgeo[1].position.z, reverse=True)
        return strings

def boundaries_MESE(geometry):
#         Side and top boundaries
#         Find the veto's side and top boundaries.
#         Parameters
#         ----------
#         geometry : I3OMGeoMap or dict(OMKey, tuple(I3OMGeo, ...))
#             IC79 or IC86 detector geometry
#         Returns

#         -------
#         sides : set(int)
#             Sequence of string numbers of the outermost strings
#         top : float
#             Depth in detector coordinates of the first DOM on the
#             deepest non-DeepCore string minus the thickness given
#             by `top_layer`

        top_layer=90.*icetray.I3Units.m,
        dust_layer=(-220.*icetray.I3Units.m,-100.*icetray.I3Units.m)
        strings = select_MESE(geometry)
        top = min(strings[s][0][1].position.z for s in strings if s <= 78)

        neighbors = collections.defaultdict(int)
        dmax = 160.*icetray.I3Units.m

        for string in strings:
            pos = strings[string][0][1].position

            for other in strings:
                if other != string:
                    opos = strings[other][0][1].position

                    # The defined maximum inter-string spacing of 160m between
                    # neighboring DOM assures the "missing" strings of the full
                    # hexagon are treated correctly.
                    if np.hypot(pos.x - opos.x, pos.y - opos.y) < dmax:
                        neighbors[string] += 1

        # The outermost strings have less than six neighbors.
        sides = [string for string in neighbors if neighbors[string] < 6]
        boundary_x=[]
        boundary_y=[]

        new_sides=[]
        new_sides[:6]=sides[0:6]
        for i in range(7,20,2):
            new_sides.append(sides[i])
        for i in range(23,20,-1):
            new_sides.append(sides[i])
        for i in range(27,23,-1):
            new_sides.append(sides[i])
        for i in range(20,5,-2):
            new_sides.append(sides[i])
        # print('newsides',new_sides)         

        # print('Boundary strings',new_sides)
        for side_string in new_sides:
            pos=strings[side_string][0][1].position
            boundary_x.append(pos.x)
            boundary_y.append(pos.y)
        boundary_x.append(boundary_x[0])
        boundary_y.append(boundary_y[0])
        # return sides, top - top_layer[0]
        return boundary_x,boundary_y


def get_surface_det_MESE(gcdFile=None):

    from icecube import MuonGun
    gcdFile=gcdFile
    bound_2D=[]
    MuonGunGCD='/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz'
    #MuonGunGCD = args.GCD
    surface_det = MuonGun.ExtrudedPolygon.from_file(MuonGunGCD, padding=50)##Build Polygon from I3Geometry (convex hull)
    f = dataio.I3File(MuonGunGCD)
    omgeo = f.pop_frame(icetray.I3Frame.Geometry)['I3Geometry'].omgeo
    surface_det_x,surface_det_y=boundaries_MESE(omgeo)#getting this from omgeo gives concave hull instead of convex hull
    x=[(surface_det_x[i],surface_det_y[i])for i in range(len(surface_det_x))]###getting only x and y
    bound_2D=mpltPath.Path(x)#Projection of detector on x,y plane

    return bound_2D, surface_det
bound_2D,surface_det = get_surface_det_MESE(gcdFile='/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz')

def boundary_check_MESE(particle1,gcdFile=None):
    ####checks if particle is inside the detector###
    gcdFile=gcdFile
    # bound_2D,surface_det = get_surface_det(gcdFile=gcdFile)
    inlimit = False
    if ((particle1.pos.z <=max(surface_det.z)) and (particle1.pos.z>=min(surface_det.z))):
        if bound_2D.contains_points([(particle1.pos.x, particle1.pos.y)]):
            inlimit=True

    return inlimit



if(args.use_grid):
    tray.Add(dataio.I3Reader, FilenameList=infiles)
else:
    tray.Add("I3Reader","reader", FilenameList=infiles)

def PrintRunID(frame):
    if frame.Has("I3EventHeader"):
        print(frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)

#tray.Add(PrintRunID, "RunID")

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return (idx-1)
    else:
        return idx


##find deposited energy in ice from MCTree, not needed for NNMFit but necessary for RF Training
##NOT USED##
def dep_energy(frame):
    #print("Deposited energy check")
    dep_casc_energy = 0
    dep_muon_energy = 0

    #find energy deposited due to secondaries
    tree = frame["I3MCTree"]
    nu = tree.primaries[0]
    child = tree.first_child(nu)

    if child.type.name in ["MuPlus", "MuMinus"]:
        print("MuGun")      ## Skips check if initial particle is muon, should be due to MuGun event 
        return True
    while child.is_neutrino:
        if tree.number_of_children(child)>0:
            nu = child
            child = tree.first_child(nu)
        else:
            break

    #first extract cascade dep energy
    neut_children = tree.get_daughters(nu)
    if len(neut_children)==1:
        return False

    if abs(neut_children[0].type)==13 or abs(neut_children[0].type)==15:         #13 for muon
        hadron = neut_children[1]
        #print(hadron)
    else:
        hadron = neut_children[0]
    had_daughters = tree.get_daughters(hadron)
    for particle in had_daughters:
        if (particle.shape.name != "Dark") and (particle.shape.name != "Primary"): ### and (particle.type in cascade_types):
            if (particle.type.name!="MuPlus") and (particle.type.name!="MuMinus"):
                if boundary_check(particle):
                    dep_casc_energy += particle.energy

    #find energy deposited from track only - potentially faulty for taus, just be aware no tests have been done
    for track in MuonGun.Track.harvest(frame['I3MCTree'], frame['MMCTrackList']):
        intersections = surface_det.intersection(track.pos, track.dir)

#        e0, e1 = track.get_energy(intersections.first), track.get_energy(intersections.second)
#        dep_muon_energy +=  (e0-e1)
#    frame["Deposited_Cascade_Energy"] = dataclasses.I3Double(dep_casc_energy)
#    frame["Deposited_Muon_Energy"]    = dataclasses.I3Double(dep_muon_energy)

        try:
            e0, e1 = track.get_energy(intersections.first), track.get_energy(intersections.second)
        except:
            print(frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
            return False
        else:
            dep_muon_energy +=  (e0-e1)
    if dep_casc_energy+dep_muon_energy == 0:
        print("Cascade", frame["IsCascade_dnn"].value)
        return False
    y_eff_dep = dep_casc_energy / (dep_casc_energy+dep_muon_energy)
    frame["Deposited_Cascade_Energy"] = dataclasses.I3Double(dep_casc_energy)
    frame["Deposited_Muon_Energy"]    = dataclasses.I3Double(dep_muon_energy)
    frame["YEff_TrueDep"] = dataclasses.I3Double(y_eff_dep)

#tray.AddModule(dep_energy, "GetDepositedEnergy")

#from ESTES_random_forest_reco import *
#tray.AddModule("Delete",'delete',Keys=['RandomForestFeatures','RandomForestEnergyEstimates','RandomForestTarget'])

## Can be used for reco'ing based on RF - collects necessary variables
#tray.AddModule(RandomForestCollect, 'collect',
#    #MillipedeName='Millipede_Free_Best_ESTES_Fit',
#    MillipedeName='Millipede_SplineMPE_TWTS',             #Name of Millipede we're basing on
#    ###MillipedeName = "MillipedeFirstLoss_TWTS",
#    NQuantiles=26,                                        #Number of quantiles for energy loss
#    FeaturesName='RandomForestFeatures',                  # Inputs for RandomForest to predict
#    TargetName=None,
#    #TargetName="RandomForestTarget",                     # True values to train Random Forest on
#    IsStartingTrack=True,                                 # Is it a Starting Track aka do muon/hadron cascade seperation
#    Cleanup=False)                                         #



#Predicts RF energies based on the pickle file

#def CascadeDummyMaker(frame):
#    frame["RandomForestEnergyEstimates"]["Cascade Deposited Energy"] = 1
#    frame["RandomForestEnergyEstimates"]["Neutrino Deposited Energy"] = 1

#tray.AddModule(RandomForestPredict, 'predict',
#    FeaturesName='RandomForestFeatures',
#    OutputName='RandomForestEnergyEstimates',
#    RandomForestPickle="/cvmfs/icecube.opensciencegrid.org/users/msilva/ESTES_files/rf_noweights.pkl")
#    #######    RandomForestPickle="/data/user/rclark/TauSimProject/rf_data/rf_NoCuts_depth20_leaf4.pkl")

## Get the reco'd y_eff from Random Forest
def y_eff_from_RF(frame):
    deposited_had_energy = frame["RandomForestEnergyEstimates"]["Cascade Deposited Energy"]
    deposited_neut_energy = frame["RandomForestEnergyEstimates"]["Neutrino Deposited Energy"]
    y_eff = deposited_had_energy/deposited_neut_energy
###    #print(y_eff)
    frame["RFYeff"] = dataclasses.I3Double(y_eff)
    #PushFrame(frame)
    return True

#tray.AddModule(y_eff_from_RF, "Y_eff_counter") 

# Get y_eff from an included millipede - make sure to check segment length
def yeff_from_milli(frame):
    #print("Millipede yeff check")
    if frame.Has("Millipede_SplineMPE_TWTS"):
        cascades = frame["Millipede_SplineMPE_TWTS"]

        #global E_total_from_losses

        non_zero_cascades = 0
        E_total_from_losses = 0
        E_Vector = []
        for cascade in cascades:
            if boundary_check_MESE(cascade): # and cascade.time >= frame["MillipedeFirstLoss_TWTS"].time:
                ####    Just summing the energy for E_total_from_losses gets you the total along the entire set of cascades
                ####    For just the energy inside the detector use frame["MillipedeDepositedEnergy_TWTS"]
                E_total_from_losses += cascade.energy   ##HERE 
                E_Vector.append(float(cascade.energy))  ## AND HERE
        #PRint(len(E_Vector))
        E_Vector = np.array(E_Vector)
        E_Vector_no_zero = E_Vector[E_Vector > 1]

        #print("Is first deposit same as Vedant says? %s" %(E_Vector_no_zero[0] == frame["MillipedeFirstLoss_TWTS"].energy))
        #print("Has FirstLoss? ", frame.Has("MillipedeFirstLoss_TWTS"))
        #print(len(E_Vector_no_zero))
        E_inDet = frame["MillipedeDepositedEnergy_TWTS"].value
        if len(E_Vector_no_zero) ==0:
            print("No non-zero reco particle")
            yeff_2point5m = 0
            yeff_5m = 0
            yeff_7point5m = 0
            yeff_10m = 0
            yeff_12point5m = 0 

            yeff_2point5m_det = 0
            yeff_5m_det = 0
            yeff_7point5m_det = 0
            yeff_10m_det = 0
            yeff_12point5m_det = 0
            
        else:
            if not (E_Vector_no_zero[0] == frame["MillipedeFirstLoss_TWTS"].energy):
                print(E_Vector_no_zero[0])
                print(frame["MillipedeFirstLoss_TWTS"].energy)
 
            yeff_2point5m = E_Vector_no_zero[0]/E_total_from_losses
            yeff_2point5m_det = E_Vector_no_zero[0]/E_inDet
            ##print("Minimum non-zero at %s, length is %s" %(np.min(np.where(E_Vector==E_Vector_no_zero[0])), len(E_Vector)))
            if np.min(np.where(E_Vector==E_Vector_no_zero[0])) +1 >= len(E_Vector):
                yeff_5m = yeff_2point5m
                yeff_5m_det = yeff_2point5m_det
            else:
                yeff_5m = (E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))]+E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+1])/E_total_from_losses
                yeff_5m_det = (E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))]+E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+1])/E_inDet
            if np.min(np.where(E_Vector==E_Vector_no_zero[0])) +2 >= len(E_Vector):
                yeff_7point5m  = yeff_5m
                yeff_7point5m_det  = yeff_5m_det
            else:
                yeff_7point5m = (E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+1] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+2])/E_total_from_losses
                yeff_7point5m_det = (E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+1] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+2])/E_inDet
            if np.min(np.where(E_Vector==E_Vector_no_zero[0]))+3 >= len(E_Vector):
                yeff_10m = yeff_7point5m
                yeff_10m_det = yeff_7point5m_det
            else:
                yeff_10m = (E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+1] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+2] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+3])/E_total_from_losses
                yeff_10m_det = (E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+1] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+2] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+3])/E_inDet
            if np.min(np.where(E_Vector==E_Vector_no_zero[0]))+4 >= len(E_Vector):
                yeff_12point5m = yeff_10m
                yeff_12point5m_det = yeff_10m_det
            else:
                yeff_12point5m = (E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+1] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+2] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+3] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+4])/E_total_from_losses
                yeff_12point5m_det = (E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+1] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+2] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+3] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+4])/E_inDet

    else:
        print("No Millipede_SplineMPE_TWTS")
        if frame["IsCascade_dnn"] == 1:
            yeff_dummy = float("nan")
        elif frame["IsCascade_dnn"] == 0:
            yeff_dummy = 0

        yeff_2point5m = yeff_dummy
        yeff_5m = yeff_dummy
        yeff_7point5m = yeff_dummy
        yeff_10m = yeff_dummy
        yeff_12point5m = yeff_dummy

        yeff_2point5m_det = yeff_dummy
        yeff_5m_det = yeff_dummy
        yeff_7point5m_det = yeff_dummy
        yeff_10m_det = yeff_dummy
        yeff_12point5m_det = yeff_dummy


    frame["yeff_2point5m"] = dataclasses.I3Double(yeff_2point5m)
    frame["yeff_5m"] = dataclasses.I3Double(yeff_5m)
    frame["yeff_7point5m"] = dataclasses.I3Double(yeff_7point5m)
    frame["yeff_10m"] = dataclasses.I3Double(yeff_10m)
    frame["yeff_12point5m"] = dataclasses.I3Double(yeff_12point5m)

    frame["yeff_2point5m_det"] = dataclasses.I3Double(yeff_2point5m_det)
    frame["yeff_5m_det"] = dataclasses.I3Double(yeff_5m_det)
    frame["yeff_7point5m_det"] = dataclasses.I3Double(yeff_7point5m_det)
    frame["yeff_10m_det"] = dataclasses.I3Double(yeff_10m_det)
    frame["yeff_12point5m_det"] = dataclasses.I3Double(yeff_12point5m_det)


tray.AddModule(yeff_from_milli, "Yeff Millipede ver")

## Get the accurate reco'd values for track or cascade
def RecoVariablesForNNMFit(frame):
    if frame["IsCascade_dnn"] == True:
        frame["RecoEnergy"] = dataclasses.I3Double(frame["ReconstructedEnergyMonopod_L5"].value)
        frame["RecoDirection"] = dataclasses.I3Double(frame["ReconstructedDirectionMonopod_L5"])
        frame["RecoParticle"] = frame["L5MonopodFit4"] 

    elif frame["IsCascade_dnn"] == False:
        frame["RecoEnergy"] = dataclasses.I3Double(frame["MillipedeDepositedEnergy_TWTS"].value)
        frame["RecoDirection"] = dataclasses.I3Double(frame["TrackFitDirection"].value)
        frame["RecoParticle"] = frame["TrackFit"]

    energy = frame["RecoEnergy"]
    if energy < 1*10**(4.5):
        if energy < 1*10**(4):
            #print("5m")
            #frame["YEff_milli"] = frame["yeff_5m"]
            frame["YEff_milli_inDet"] = frame["yeff_5m_det"]
        else:
            #print("7.5m")
            frame["YEff_milli"] = frame["yeff_7point5m"]
            frame["YEff_milli_inDet"] = frame["yeff_7point5m_det"]
    else:
        if energy < 1*10**(5):
            #print("10m")
            frame["YEff_milli"] = frame["yeff_10m"]
            frame["YEff_milli_inDet"] = frame["yeff_10m_det"]
        else:
            #print("12.5m")
            frame["YEff_milli"] = frame["yeff_12point5m"]  
            frame["YEff_milli_inDet"] = frame["yeff_12point5m_det"]

tray.AddModule(RecoVariablesForNNMFit, "RecoVarGetter")

Taus = [dataclasses.I3Particle.TauMinus, dataclasses.I3Particle.TauPlus]
Mus = [dataclasses.I3Particle.MuMinus, dataclasses.I3Particle.MuPlus]

#sys.path.append('/home/rclark/scripts/RF_and_NNMScripts')
from GetI3Parent import *            

def trace_energy_2(frame):
    
    energy = frame["RecoEnergy"]
    if energy < 1*10**(4.5):
        if energy < 1*10**(4):
            dist = 5
        else:
            dist = 7.5
    else:
        if energy < 1*10**(5):
            dist = 10
        else:
            dist = 12.5

    
    event = frame["I3EventHeader"].event_id
    MCTree = "I3MCTree"
    if frame.Has(MCTree):
        E_close = 0
        E_far = 0
        
        tree = frame[MCTree]
        primary_nu, child_lep = FindTrueParent(tree)
        vertex = child_lep.pos
        #print(primary_nu.type, primary_nu.pos)
        children = tree.children(primary_nu)
        for child in children:       
            daughter_energy = 0
            for x in tree.children(child):
                if x.type in cascade_types and boundary_check_MESE(x):
                    dist_from_vert = (x.pos-vertex).magnitude
                    if dist_from_vert < dist:                        
                        E_close += x.energy
                    else:
                        E_far += x.energy
                    daughter_energy+= x.energy
                elif x.type in Mus:
                    Mu_children = frame[MCTree].get_daughters(x)
                    for decay_child in Mu_children:
                        if decay_child.type in cascade_types and boundary_check_MESE(decay_child):
                            dist_from_vert = (decay_child.pos-vertex).magnitude
                            if dist_from_vert < dist:
                                E_close += decay_child.energy
                            else:
                                E_far += decay_child.energy
                            daughter_energy+= decay_child.energy
        frame["y_eff_true"] = dataclasses.I3Double(E_close/(E_close+E_far))
        frame["E_close"] = dataclasses.I3Double(E_close)
        frame["E_far"] = dataclasses.I3Double(E_far)
tray.AddModule(trace_energy_2, "Trace Energy")                    

def AngularResolution(frame):
    if "MCPrimary" in frame:
        primary=frame['MCPrimary']
        reco=frame['RecoParticle']
        opening_angle=(np.arccos(reco.dir*primary.dir))
        frame['OpeningAngle']=dataclasses.I3Double(opening_angle)


def NeutrinoDepth(frame):
    if "NFiles" not in frame:
        frame["NFiles"]=dataclasses.I3Double(len(infiles))
    if "MCVetoMuonInjectionInfo" in frame:
        if "muon_5_energy" in frame["MCVetoMuonInjectionInfo"]:
            print('too many injection muons')
            return False
    # if 'ShowerNeutrino' in frame and 'PrimaryDepthMC' not in frame:
    if 'I3MCTree' in frame and 'PrimaryDepthMC' not in frame:
        # neutrino=frame['ShowerNeutrino']
        neutrino = None
        mctree = frame['I3MCTree']
        for p in mctree:
            depth = mctree.depth(p)
            if (depth == 0):
                if neutrino is None and len(mctree.get_daughters(p))>0:
                    neutrino = p
                    # print(neutrino)


        intersection=surface_det.intersection(neutrino.pos, neutrino.dir)#points of intersection
        z_inter=neutrino.pos.z-intersection.first*np.cos(neutrino.dir.zenith)
        depth=1948.07-z_inter
        frame["PrimaryDepthMC"]=dataclasses.I3Double(depth)
        print('filenum:',frame['I3EventHeader'].run_id,' Event:',frame['I3EventHeader'].event_id)
        # neutrinoparent=frame['NeutrinoParent'].pdg_encoding
        # print(neutrinoparent)
        # frame["PrimaryMass"]=dataclasses.I3Double(mass_dict[str()])


iformat = 'hdf5'
table_keys = ['VisibleEnergyMC','ReconstructedEnergyMonopod_L5',
                'ReconstructedDirectionMonopod_L5','ReconstructedTypeMonopod_L5','L5MonopodFit4',
                'MuEXAngularEnergy','MuEXAngularDirection',
                'TrackFitEnergy','TrackFitDirection','TrackLength','TrackFit',
                'TrackFit_AvgDistQ','L5MonopodFit4_AvgDistQ','PrimaryFlavour','PrimaryDepthMC','Interaction_Type',
                'MillipedeDepositedEnergy_TWTS',
                'OneWeight','MuonWeight','MuonWeight2',
                'Weight_Honda2004','Weight_Hoerandel5','Weight_GaisserH4a',
                'PrimaryEnergyMC','PrimaryZenithMC',"MCPrimaryType","MCPrimary",
                "PolyplopiaPrimary", "I3MCWeightDict","CorsikaWeightMap","PrimaryMass","NeutrinoParent",
                'IsUpgoingMuon_L4',
                'MuonWeight_corrected',
                'VetoTrackMarginTWTSInIceL5Top','VetoTrackMarginTWTSInIceL5Side',
                'VetoTrackMarginMilliTWTSInIceTop','VetoTrackMarginMilliTWTSInIceSide',
                'VetoLayer0','VetoLayer1','VetoLayerQTot',
                'DNN_classification_test_TWTS_base',
                'Muon_Energy_L1','Muon_Energy_L2','Muon_Energy_L3','Muon_Energy_L4','Muon_Energy_L5','Total_Muon_energy',
                'Muon_L1','Muon_L2','Muon_L3','Muon_L4','Muon_L5',
                'Muon_L1_Depth','Muon_L2_Depth','Muon_L3_Depth','Muon_L4_Depth','Muon_L5_Depth',
                'MuonEnergy','MuonMultiplicity','MuonMultiplicity_10','MuonMultiplicity_100','MuonMultiplicity_200',
                'ShowerNeutrino','max_depo_energy','BrightestMuon','HEMuon','BrightestMuonDepth','Energy_HEMuonAtDepth','HEMuonDepth',
                'BrightestMuonAtDepth','Energy_BrightestMuonAtDepth',
                'IsHese','IsHESE_jvs',"IsHESE_ck",'IsMESEL3',
                'IsCascade_dnn','IsTrack_dnn',"NFiles",
                'HomogenizedQTot','HomogenizedQTot_TWTS','HomogenizedQTot_toposplit','Millipede_RobustFit_TrackFit',
                'IsStartingEvent_L4','UpgoingMuon_CoincCut','Filenum','FilterMask',
                'IsCascade_recoL4','IsTrack_recoL4','SPEFit4_rlogL' ,'SPEFitSingle_rlogL','SPEFit2_rlogL','TrackFitFitParams','OpeningAngle',
                'IsVetoTrack','IsVetoTrack_New',"I3EventHeader",'SPEFit4_offlineFitParams' ,'SPEFitSingle_offlineFitParams','SPEFit2_offlineFitParams',
                'IsCascade_true','IsTrack_true','MCVetoMuonInjectionInfo','Side','Top',
                'RecoEnergy','RecoParticle','RecoDirection','SnowstormParameterDict','AtmNeutrinoPassingFraction','AtmNeutrinoPassingFraction_New',
                "NEvents", "RandomForestEnergyEstimates","RFYeff", "YEff_milli", "YEff_milli_inDet", "YEff_TrueDep", "yeff_2point5m", "yeff_5m", "yeff_7point5m", "yeff_10m", "yeff_12point5m",
                "yeff_2point5m_det", "yeff_5m_det", "yeff_7point5m_det", "yeff_10m_det", "yeff_12point5m_det", "y_eff_true", "E_close", "E_far",]



if len(outfilebase)==0:
    outfilebase="test"
#outfile = outfilebase+".i3.zst"
if(args.use_grid):
    outfile = my_grid+outfilebase
else:
    outfile = outfilebase
#print (outfile)

tray.Add(
    hdfwriter.I3HDFWriter,
    SubEventStreams=["InIceSplit", "topological_split"],
    keys = table_keys,
    output=outfile+".hdf5",
    )


tray.AddModule('I3Writer', 'writer',
    DropOrphanStreams=[icetray.I3Frame.DAQ],
    Streams=[  icetray.I3Frame.DAQ, icetray.I3Frame.Physics,icetray.I3Frame.Simulation,
              icetray.I3Frame.Stream('M')],
    filename=outfile+".i3.zst")


if keep_gcd:
    tray.AddModule("I3Writer", Filename=outfile+".i3.zst",
        Streams=[icetray.I3Frame.Geometry,
            icetray.I3Frame.Calibration,
            icetray.I3Frame.DetectorStatus,
            icetray.I3Frame.DAQ,
            icetray.I3Frame.Physics,
            icetray.I3Frame.Simulation,
            icetray.I3Frame.Stream('M')],
        DropOrphanStreams=[icetray.I3Frame.DAQ,
            icetray.I3Frame.Geometry,
            icetray.I3Frame.Calibration,
            icetray.I3Frame.DetectorStatus])
else:
    tray.AddModule("I3Writer", Filename=outfile+".i3.zst",
        Streams=[  icetray.I3Frame.DAQ, icetray.I3Frame.Physics,icetray.I3Frame.Simulation,
                 icetray.I3Frame.Stream('M')],
         DropOrphanStreams=[icetray.I3Frame.Geometry,
            icetray.I3Frame.Calibration,
            icetray.I3Frame.DetectorStatus,
            icetray.I3Frame.DAQ])

tray.AddModule("TrashCan", "thecan")
tray.Execute()
tray.Finish()

stop_time = time.asctime()

print ('Started:', start_time)
print ('Ended:', stop_time)

