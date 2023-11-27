#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT: combo/V00-00-03
"""
Collect random forest features and targets and write to an hdf5 file
"""
from I3Tray import *
from icecube import dataclasses, icetray, MuonGun, millipede
from icecube.dataclasses import I3Particle
from icecube import hdfwriter
from icecube.icetray import traysegment
####from ESTES_random_forest_reco_forEstes import *
#from ESTES_random_forest_reco import *
from icecube.rootwriter import I3ROOTWriter
from optparse import OptionParser
import numpy as np
import os
from glob import glob
from icecube.simprod import segments
import time
import math

from datetime import datetime
import sys,gzip,pickle,urllib
#print (uname())
#from icecube import simclasses
from icecube import dataclasses,icetray,dataio,phys_services,VHESelfVeto,StartingTrackVeto, hdfwriter, recclasses, simclasses
from icecube.dataclasses import I3Particle
from numpy import sign as sign
#from numba import jit
import collections


start_time = time.asctime()
print ('Started:', start_time)

global event_count; event_count = 0;


parser = OptionParser()
parser.allow_interspersed_args = True
parser.add_option("-i", "--infile", default="21217",
                  dest="infile", help="input file did", type=str)
parser.add_option("-g", "--geofile", default="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz", type=str,
                  dest="gcd", help="GCD file")

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

# parse cmd line args,
(opts,args) = parser.parse_args()
file_no = opts.infile.split("_")[-1]
print(file_no)

if int(file_no) in [22010,22011,22012,22013,22014,22015,22016,22017,22018]:
#    infiles = [opts.gcd]+sorted(glob("/data/user/vbasu/CONDOR_output/SnowStormNuGen/L4_DNN/L4_NewIce_DNN_0%s_*.i3.zst" %(file_no)))
#    infiles = [opts.gcd]+sorted(glob("/data/ana/MESE/NuGen/SnowStorm/L5_newFiducial/L5_NewIce_%s_000000.i3.zst" %(file_no)))
    infiles = [opts.gcd]+sorted(glob("/data/ana/MESE/NuGen/SnowStorm/L5_newFiducial/L5_NewIce_%s_*.i3.zst" %(file_no)))
elif int(file_no) in [21217,21218,21219]:
#    infiles= [opts.gcd]+sorted(glob("/data/user/rclark/CONDOR_output/MESE_L4_PropNu*/L4_NewIce_0%s_*.i3.zst" %(file_no)))i
    infiles = [opts.gcd]+sorted(glob("Tracks_RemoveOutDetector_%s.i3.zst" %(file_no)))
elif int(file_no) in [21813,21814,21938]:
    infiles = [opts.gcd]+sorted(glob("/data/ana/Muon/ESTES/ESTES_2019/ESTES_data/nugen_numu_%s_p0=0.0_p1=0.0/*step4.i3*" %(file_no)))

else:
    infiles = opts.gcd
    print("No Files selected")

##infiles = [opts.gcd]+sorted(glob("/data/ana/Muon/ESTES/ESTES_2019/ESTES_data/nugen_numu_21813_p0=0.0_p1=0.0/*step4.i3*"))

print(infiles[:10])



###Define a surface boundary check
import matplotlib.path as mpltPath

def select(geometry):
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

def boundaries(geometry):
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
        strings = select(geometry)
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

def get_surface_det(gcdFile=None):

    from icecube import MuonGun
    gcdFile=gcdFile
    bound_2D=[]
    MuonGunGCD='/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz'
    surface_det = MuonGun.ExtrudedPolygon.from_file(MuonGunGCD, padding=0)##Build Polygon from I3Geometry (convex hull)
    f = dataio.I3File(MuonGunGCD)
    omgeo = f.pop_frame(icetray.I3Frame.Geometry)['I3Geometry'].omgeo
    surface_det_x,surface_det_y=boundaries(omgeo)#getting this from omgeo gives concave hull instead of convex hull
    x=[(surface_det_x[i],surface_det_y[i])for i in range(len(surface_det_x))]###getting only x and y
    bound_2D=mpltPath.Path(x)#Projection of detector on x,y plane

    return bound_2D, surface_det
bound_2D,surface_det = get_surface_det(gcdFile='/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz')

def boundary_check(particle1,gcdFile=None):
    ####checks if particle is inside the detector###
    gcdFile=gcdFile
    # bound_2D,surface_det = get_surface_det(gcdFile=gcdFile)
    inlimit = False
    if ((particle1.pos.z <=max(surface_det.z)) and (particle1.pos.z>=min(surface_det.z))):
        if bound_2D.contains_points([(particle1.pos.x, particle1.pos.y)]):
            inlimit=True

    return inlimit

### We now have a boundary check



tray = I3Tray()
tray.AddModule('I3Reader', 'reader',
               FilenameList = infiles)

def particle_type_finder(frame):
    if frame.Has("IsTrack_dnn"):
        if frame["IsTrack_dnn"] == True:     ###CURRENTLY SET FOR TRACKS
            #print("START: Track")
            return True
        else:
            #print("START: Cascade")
            return False
tray.Add(particle_type_finder)


##def TauMCTruth(tray,name):
#    """
#    This checks and generates the MC truth of tau events. 
#    MCTruth_track=True and track events and =False are cascade/double cascade events
#    """
def TauGeneratesMuon(frame, MCTreeName = "I3MCTree", OutBool = "HasMuon", OutgoingMuonName = "OutgoingMuon"):
    print("ASWATHI CODE")
    if frame.Has(MCTreeName):
        printval=False
        frame['MCTruth_track']=icetray.I3Bool(False)
        for p in frame[MCTreeName].get_primaries():
            print("Parent type: %s" %p.type)
            if (frame["IsCascade_dnn"].value==False):
                printval=True
            if printval==True:
                print("cascade_dnn", frame["IsCascade_dnn"], frame["I3EventHeader"].event_id)
            if (p.type != dataclasses.I3Particle.NuTau and p.type != dataclasses.I3Particle.NuTauBar):
                continue
            for c in frame[MCTreeName].children(p):
                if printval==True:
                    print("Child type", c.type)
                if c.type == dataclasses.I3Particle.TauMinus or c.type == dataclasses.I3Particle.TauPlus:
                        print("Daughters of lepton", [d.type for d in frame[MCTreeName].get_daughters(c)])
                        if len(frame[MCTreeName].get_daughters(c))==0:
                            print("tau has no daughters and this is a track")
                            del frame['MCTruth_track']
                            frame['MCTruth_track']=icetray.I3Bool(True)
                for d in frame[MCTreeName].get_daughters(c):
                    if d.type == dataclasses.I3Particle.MuPlus or d.type == dataclasses.I3Particle.MuMinus:
                        if printval==True:
                           print(c.type, d.type, d.energy, d.energy/p.energy, d.energy/c.energy)
                        if (d.energy/c.energy>0.1 or d.energy > 150):
                            del frame['MCTruth_track']
                            frame['MCTruth_track']=icetray.I3Bool(True)
                            if printval==True:
                                 print("this could be a track")

    print("END ASWATHI CODE")
    return True
#tray.Add(TauGeneratesMuon)

def pre_cut(frame):
    global event_count
    # numu CC or nutau CC only
    wd = frame['I3MCWeightDict']
    if not (wd['InteractionType']==1): ## and (abs(wd['PrimaryNeutrinoType'])!=14 or abs(wd['PrimaryNeutrinoType'])!=16)):
        print(wd['InteractionType'], wd['PrimaryNeutrinoType'])
        return False

    # Find in-ice neutrino, muon
    tree = frame["I3MCTree_preMuonProp"]
    nu = tree.primaries[0]
    if not nu.is_neutrino:
        return False
    child = tree.first_child(nu)
    while child.is_neutrino:
        try:
            nu = child
            child = tree.first_child(nu)
        except:
            print("Bizarre no child")
            print(frame['I3EventHeader'])
            return False
 
    muon = child
    #print(muon)

    # Find intersection points of muon track with detector volume
    ip = phys_services.Surface.intersection(surface_det, muon.pos, muon.dir);
    ip = [ip.first, ip.second];
    
    ###if len(ip)>0:
    ###    length_before = ip[0]
    ###    length_after = ip[-1]
    ###    # Remove if vertex outside detector
    ###    if (length_before > 0) or (length_after < 0):
    ###        print("Outside detector")
    ###        return False
   ### else:
    ###    print("IP < 0")
    ###    return False
    
    # Remove badly reconstructed events 
    track = frame['MillipedeFirstLoss_TWTS']            ## Change here for ESTES/MESE
    #track = frame["Millipede_Free_Best_ESTES_Fit"]
    ###if np.degrees(np.arccos(track.dir*muon.dir)) > 5:
    ###    print("Badly reco'd event")
    ###    return False
    #print("Passes Pre Cuts")
    event_count+=1
    return True  
tray.AddModule(pre_cut, 'pre_cut')

def BrokenDaughterCheck(frame):
    ### Checks for regenerated neutrino and compensates by digging through the MC Tree
    tree_preprop = frame["I3MCTree_preMuonProp"]
    while not boundary_check(child):


        if tree_preprop.number_of_children(child) == 0:
            #tree.erase_children(parent)
            child = tree_preprop.parent(nu)
            nu = tree_preprop.parent(child)
            GC = tree_preprop.first_child(child)
            print("Fail branch: ", GC.type.name)
            tree_preprop.erase(GC)

        for GC in tree_preprop.get_daughters(child):
            for GGC_lep in tree_preprop.get_daughters(GC):
                if GC.is_neutrino and not tree_preprop.number_of_children(GC)==0 and GGC_lep.type.name in ["EPlus", "EMinus", "MuPlus", "MuMinus", "TauPlus", "TauMinus"]:
                    nu = GC
                    child = GGC_lep

        #           while child.is_neutrino:
         #              if tree.number_of_children(child)>0:
          #                 nu = child
           #                child = tree.first_child(nu)
            #           else:
             #              break
#
 #               elif GC.is_neutrino and GGC_lep.is_neutrino:
  #                  

def FindingTheDaughtersWhichActuallyInteract(frame):
    tree_preprop = frame["I3MCTree_preMuonProp"]
    nu = tree_preprop.primaries[0]
    child = tree_preprop.first_child(nu)

    while (not boundary_check(child)) or (not abs(child.type) in [11,13,15]):
        if tree_preprop.number_of_children(child) == 0:
            child = tree_preprop.parent(child)
            GC = tree_preprop.first_child(child)
            tree_preprop.erase(GC)
        else:
            child = tree_preprop.first_child(child)

    print("Final child is: ", child.type.name, child.pos)

#tray.AddModule(FindingTheDaughtersWhichActuallyInteract)

#find deposited energy
#tray.AddModule("Delete", 'delete deposited', Keys=["Deposited_Cascade_Energy", "Deposited_Muon_Energy", "True_Deposited_Yeff", "Cascade_energy", "Muon_energy", "MESEWeight", "numuWeight", "HESEWeight"])
    
def dep_energy(frame):
    dep_casc_energy = 0
    dep_muon_energy = 0
    
    #find energy deposited due to secondaries
    tree = frame["I3MCTree"]
    nu = tree.primaries[0]
    child = tree.first_child(nu)
    while child.is_neutrino:
        if tree.number_of_children(child)>0:
            nu = child
            child = tree.first_child(nu)
        else:
            break
    

    tree_preprop = frame["I3MCTree_preMuonProp"]

    while (not boundary_check(child)) or (not abs(child.type) in [11,13,15]):
        if tree_preprop.number_of_children(child) == 0:
            try:
                child = tree_preprop.parent(child)

            except:
                 return False
            GC = tree_preprop.first_child(child)
            tree_preprop.erase(GC)
        else:
            child = tree_preprop.first_child(child)

    nu2 = tree_preprop.parent(child)
    if nu.is_neutrino:
        nu = nu2  
    #print("Final child is: ", child.type.name, child.pos)
    
    #first extract cascade dep energy
    neut_children = tree.get_daughters(nu)
    if len(neut_children)==1:
        print("One child")
        print(neut_children[0].type.name)
        print(neut_children[0].pos)
        return False

    if abs(neut_children[0].type) in [11,13,15]:         #13 for muon, 15 for tau
        hadron = neut_children[1]
        #print(hadron)
    else:
        hadron = neut_children[0]

    int_vertex = hadron.pos
    print("Interaction happening due to ", nu.type.name, "at",  hadron.type.name, hadron.pos)
    max_cascade_extent = 0
    average_cascade_extent = np.array([])

    had_daughters = tree.get_daughters(hadron)
    for particle in had_daughters:
        if (particle.shape.name != "Dark") and (particle.shape.name != "Primary"): ### and (particle.type in cascade_types):
            if (particle.type.name!="MuPlus") and (particle.type.name!="MuMinus"):
                #print(particle.type.name, particle.pos)
                if boundary_check(particle):
                    dep_casc_energy += particle.energy
                    production_vertex = particle.pos
                    extent = production_vertex - int_vertex
                    #print(extent)
                    distance_from_vertex = np.linalg.norm(extent)
                    average_cascade_extent = np.append(average_cascade_extent, distance_from_vertex)
                    if distance_from_vertex > max_cascade_extent:
                        max_cascade_extent = distance_from_vertex
    print("%s m" %(np.mean(average_cascade_extent)))
    frame["Max_Cascade_Extent"] = dataclasses.I3Double(max_cascade_extent)
    frame["Average_Cascade_Extent"] = dataclasses.I3Double(np.mean(average_cascade_extent))
    frame["Cascade_Lengths"] = dataclasses.I3VectorDouble(average_cascade_extent)
    frame["FinalLepton"] = dataclasses.I3Particle(child)
#    find energy deposited from track only
    for track in MuonGun.Track.harvest(frame['I3MCTree'], frame['MMCTrackList']):
        intersections = surface_det.intersection(track.pos, track.dir)       
         
        try:       
            e0, e1 = track.get_energy(intersections.first), track.get_energy(intersections.second)
        except:
            print(frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
            return False
        else:
            dep_muon_energy +=  (e0-e1)
#    frame["Deposited_Cascade_Energy"] = dataclasses.I3Double(dep_casc_energy)
#    frame["Deposited_Muon_Energy"]    = dataclasses.I3Double(dep_muon_energy)
#    if dep_muon_energy==0 and dep_casc_energy==0:
#        return False

#    frame["True_Deposited_Yeff"] = dataclasses.I3Double((dep_casc_energy)/(dep_casc_energy+dep_muon_energy))

tray.Add(dep_energy) ##,Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics])

#tray.AddModule("Delete",'delete',Keys=['RandomForestFeatures','RandomForestEnergyEstimates','RandomForestTarget',])
'''
tray.AddModule(RandomForestCollect, 'collect',
               MillipedeName='Millipede_SplineMPE_TWTS',
               ###MillipedeName="Millipede_Free_Best_ESTES_FitParams",  ##Useless line, do not use
               #MillipedeName="Millipede_Free_Best_ESTES_Fit",
               NQuantiles=26,
               FeaturesName='RandomForestFeatures',
               TargetName='RandomForestTarget',
               IsStartingTrack=True,
               Cleanup=True)
'''
def simple_weight(frame):
    ow = frame['I3MCWeightDict']['OneWeight']
    e = frame['I3MCWeightDict']['PrimaryNeutrinoEnergy']
    nev = frame['I3MCWeightDict']['NEvents']
    meseflux = (2.06*10**-18)*((e/(100000.))**-2.46)
    frame['MESEWeight'] = dataclasses.I3Double(ow*meseflux/(nev/2))
    heseflux = (2.15*10**-18)*((e/(100000.))**-2.89)
    frame['HESEWeight'] = dataclasses.I3Double(ow*heseflux/(nev/2))
    numuflux = (1.44*10**-18)*((e/(100000.))**-2.28)
    frame['numuWeight'] = dataclasses.I3Double(ow*numuflux/(nev/2))
    return True
#tray.AddModule(simple_weight, 'weight')

output = "/home/rclark/scripts/RF_and_NNMScripts/HadronicCascadeExtent_%s_%sfiles.hdf5" %(str(file_no), str(len(infiles)-1))
print(output)

tray.Add(
    hdfwriter.I3HDFWriter,
    SubEventStreams=["InIceSplit", "topological_split"],
    keys=["I3MCWeightDict", "HomogenizedQTot", "MillipedeFirstLoss_TWTS", "Millipede_SplineMPE_TWTSFitParams", "I3EventHeader", "IsCascade_dnn", "Max_Cascade_Extent", "Average_Cascade_Extent", "Cascade_Lengths",
           "ReconstructedEnergyMonopod_L5", "MillipedeDepositedEnergy_TWTS", "True_Deposited_Yeff", "AtmNeutrinoPassingFraction", "MuonWeight", "Filenum", "ReconstructedDirectionMonopod_L5", "TrackFitDirection",
           "Deposited_Cascade_Energy", "Deposited_Muon_Energy", "MESEWeight", "HESEWeight", "numuWeight", "Millipede_Free_Best_ESTES_Fit", "Cascade_energy", "Muon_energy", "YEff_milli_2.5m", "YEff_milli_5m", "FinalLepton",
          ],
    output=output,
    )

'''
tray.AddModule("I3Writer", Filename="HadronicCascadeExtent_%s.i3.zst" %(str(file_no)),
        Streams=[icetray.I3Frame.DAQ,
            icetray.I3Frame.Physics],
        DropOrphanStreams=[icetray.I3Frame.Geometry,
            icetray.I3Frame.Calibration,
            icetray.I3Frame.DetectorStatus,
            icetray.I3Frame.DAQ])
'''
tray.Execute()
tray.Finish()


#outfile = "/data/user/rclark/TauSimProject/rf_data/rf_inputs_testESTES_%s.npy" %opts.infile
#if(len(output_array) > 0):
#    np.save(outfile,output_array)

print("Done!")

print(event_count)
stop_time = time.asctime()

print ('Started:', start_time)
print ('Ended:', stop_time)
