#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-start
#METAPROJECT /cvmfs/icecube.opensciencegrid.org/users/vbasu/meta-projects/combo3/build

#####
# !/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
####
## METAPROJECT: combo/V00-00-03

## RUN ME ON NPX

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
import pandas as pd
#from numba import jit
import numpy as np
import time

from scipy.optimize import curve_fit
from scipy.stats import chi2
from scipy.special import erfc
def chi2cdf(x,df1,loc,scale):
    func = chi2.cdf(x,df1,loc,scale)
    return func
def erfc_fit(x,scale,loc,sigma):
    x_scale=(x-loc)/(sigma*np.sqrt(2))
    func = scale*(2-erfc(x_scale))
    return func
def line_fit(x,grad, intc):
    func = grad*x + intc
    return func
os.environ["PATH"] += os.pathsep + "/home/user/rclark/scripts/RF_and_NNMScripts/"
sys.path.append('/home/rclark/scripts/RF_and_NNMScripts/')


import os
import numpy as np
from scipy.integrate import simps
from scipy.integrate import quad
from scipy import interpolate
from scipy.interpolate import interp1d
from collections import Iterable
from copy import copy

start_time = time.asctime()
print("Starting: ", start_time)

desc="icetray script to extract info"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("-i","--infolder",dest="infolder", type=str,default="022010",nargs="+", help="[I]nfolder of files")
parser.add_argument("-o","--outfile",dest="outfile",type=str,default="testOfCombo", help="[I]nfolder of files")
#parser.add_argument("-b", "--is_burn", dest="is_burn", help="Are we dealing with burn?")
args = parser.parse_args()
filenum = args.infolder[0]
#is_burn = args.is_burn

'''
if filenum == "NuE":
    filedir = "/data/ana/MESE/NuGen/SnowStorm/L5_withYEff"
    infiles = sorted(glob.glob(filedir+"/L5_NewIce_02201[3,4,5]*.i3.zst"))
elif filenum == "NuMu":
    filedir = "/data/ana/MESE/NuGen/SnowStorm/L5_withYEff"
    infiles = sorted(glob.glob(filedir+"/L5_NewIce_02201[0,1,2]*.i3.zst"))
elif filenum == "NuTau":
    filedir = "/data/ana/MESE/NuGen/SnowStorm/L5_withYEff"
    infiles = sorted(glob.glob(filedir+"/L5_NewIce_02201[6,7,8]*.i3.zst"))
elif filenum == "MuonGun":
    filedir = "/data/ana/MESE/MuonGun/L5_withYEff"
    infiles = sorted(glob.glob(filedir+"/L5_NewIce_02131*.i3.zst"))
elif filenum == "Burn":
    filedir = "/data/ana/MESE/Burn/*/L5_withYEff"
    infiles = sorted(glob.glob(filedir+"/Burn_L5_*.i3.zst"))
'''

if int(filenum) in [22010, 22011, 22012, 22013, 22014, 22015, 22016, 22017, 22018]:
    #filedir = "/data/ana/MESE/NuGen/SnowStorm/L5_withYEff_newBC"
    filedir = "/data/ana/MESE/NuGen/SnowStorm/L5_withYEff_trueYEff"
    infiles = sorted(glob.glob(filedir+"/L5_NewIce_%s*.i3.zst" %(filenum)))
    is_burn = 0
elif int(filenum) in [21315, 21316, 21317, 21318, 21319]:
    filedir = "/data/ana/MESE/MuonGun/L5_withYEff_newBC"
    infiles = sorted(glob.glob(filedir+"/L5_NewIce_%s_*.i3.zst" %(filenum)))
    is_burn = 0
elif int(filenum) in [2016, 2017]:
    filedir = "/data/ana/MESE/Burn/%s/L5_withYEff_newBC" %filenum
    infiles = sorted(glob.glob(filedir+"/Burn_L5_%s*.i3.zst" %(filenum)))
    is_burn = 1
elif int(filenum) in [2011, 2012, 2013, 2014, 2015, 2018, 2019, 2020, 2021]:
    is_burn = 1
    filedir = "/data/ana/MESE/Stage1/%s/L5_withYEff_newBC" %filenum
    infiles = sorted(glob.glob(filedir+"/Burn_L5_%s*.i3.zst" %(filenum)))


### FOR SMALL SAMPLE EXPERIMENTING - USE ONLY FIRST 1500 files or whatever's closest ###
#ilimit_number = {"022010":1500, "022011":1500, "022012":1500, "022013":748, "022014":1500, "022015":1500, "022016":898, "022017":1500, "022018":1500,
#                "021315":13499, "021316":38446, "021317":16458, "021318":43234, "021319":48491}
#infiles = infiles[:(limit_number[filenum])]

print(infiles[:5], infiles[-5:])
#infiles = infiles[:1000]
number_of_files = len(infiles)
print(number_of_files)

sys.path.append('/cvmfs/icecube.opensciencegrid.org/users/rclark/')
from GetI3Parent import *
#from home.rclark.scripts.RF_and_NNMScripts.GetI3Parent import *

tray = I3Tray()
tray.Add("I3Reader","reader", FilenameList=infiles)

tray.AddModule("Delete",'delete',Keys=['AtmNeutrinoPassingFraction_New', "AtmNeutrinoPassingFraction", "AtmNeutrinoPassingFraction_New_IntermLower", "AtmNeutrinoPassingFraction_New_a", "AtmNeutrinoPassingFraction_New_b", "AtmNeutrinoPassingFraction_New_c",
                 "AtmNeutrinoPassingFraction_New_Pr", "AtmNeutrinoPassingFraction_New_Pr_Upper", "AtmNeutrinoPassingFraction_New_Pr_Lower", "AtmNeutrinoPassingFraction_New_Pr_IntermUpper",
                "AtmNeutrinoPassingFraction_New_Pr_IntermLower", "AtmNeutrinoPassingFraction_New_Pr_a", "AtmNeutrinoPassingFraction_New_Pr_b", "AtmNeutrinoPassingFraction_New_Pr_c"])
###

#tray.AddModule(CrossSectionBounds, "Cross Section Bounds")
###
from MESE_CrossSectionFns import *

def dsdy_model(y, lmd, ybar):   ##Parametrised model for d(sigma)/dy given some lambda and <y>
    n = 0.5 * (lmd+1) * ((ybar * (lmd+3)) - lmd)
    eps = (-0.5 * (lmd+2) * (lmd+3)) * ((ybar * (lmd+1)) - lmd)/((ybar * (lmd+3)) - lmd)
    return n * (1 + eps * (1-y)**2) * np.power(y, lmd-1)

def return_weights(E, y, dsdy_interp, avg_y_interp):
    proton_mass = 0.938
    ymin = 1 / (2 * proton_mass * E)
    if ymin > y:
        print("Lower than physically possible")
    
    ## Fit lambda at E
    ys = np.linspace(0.001, 1, 1001)
    dsdy_ = dsdy_interp(np.log10(E), np.log10(ys))               # Array of "true" cross sections
    xsec = lambda y0: dsdy_interp(np.log10(E), np.log10(y0))     # Function that returns cross section at given E and y
    sigma_, err = quad(xsec, ymin, 1, epsabs=1e-10)       # Calculates integrated sigma at given E
    ybar = avg_y_interp(np.log10(E))                       # Calculates average y or y_bar at given E

    dsdy_model_fixed_ybar = lambda y, lmd: dsdy_model(y, lmd, ybar)         # Model to predict

    popt, pcov = curve_fit(dsdy_model_fixed_ybar, ys, dsdy_/sigma_,         # Popt in this case is lambda
                           bounds=([-np.inf], [np.inf]),
                           p0=(0.5),
                           method='trf',
                           ftol=1e-15)

    scale_val = scale_factor_interp(np.log10(E))
    scale_val = 1.1
    print(E, scale_val)

    #dsdy_0 = dsdy(np.log10(E), np.log10(y))
    dsdy_0 = dsdy_model(y, ybar=ybar, *popt)
    dsdy_lower = dsdy_model(y, ybar=ybar/scale_val, *popt)
    dsdy_upper = dsdy_model(y, ybar=ybar*scale_val, *popt)
    
    return dsdy_lower/dsdy_0, dsdy_upper/dsdy_0

def get_weight_bounds(E, y, int_neutrino, int_type):
    if int_type == 1.0 and int_neutrino > 0:
        print("NuCC")
        neut_type = "nu"
        current = "CC"
    elif int_type == 2.0 and int_neutrino > 0:
        print("NuNC")
        neut_type = "nu"
        current = "NC"
    elif int_type == 1.0 and int_neutrino < 0:
        print("NuBarCC")
        neut_type = "nubar"
        current = "CC"
    elif int_type == 2.0 and int_neutrino < 0:
        print("NuBarNC")
        neut_type = "nubar"
        current = "NC"
        
    else:
        print("Probably Glashow, assume no change for now")
        return 1, 1
    
    return return_weights(E, y, dsdy_index[neut_type][current], avg_y_index[neut_type][current])

import json
#with open("/data/ana/Diffuse/NNMFit/resources/inelasticity_tables/NuGen_xsection_fits.json") as f:
with open('/home/rclark/xsection_fits_cluster_NewReducedE.json') as f:
    data = json.load(f)
    
avg_y_index = {}
dsdy_index = {}

for neut_type in ["nu", "nubar"]:
    avg_y_index[neut_type] = {}
    dsdy_index[neut_type] = {}
    for int_type in ["CC", "NC"]:
        
        E = np.array(list(data[neut_type][int_type].keys()))

        E_int = [float(energy) for energy in E]
        ave_y = [float(data[neut_type][int_type][energy]["<y>"]) for energy in E]
        avg_y_interp = interp1d(E_int, ave_y)

        scale_value = np.array([energy/50+26/25 for energy in E_int])
        scale_factor_interp = interp1d(E_int, scale_value)

        energy_grid_v = []
        y_grid_v = []
        x_sec_grid_v = []

        for energy_val in E:
            for val in data[neut_type][int_type][energy_val]["yrange"]:
                energy_grid_v = np.append(energy_grid_v, float(energy_val))
            y_grid_v = np.concatenate((y_grid_v, np.log10(np.array(data[neut_type][int_type][energy_val]["yrange"]))))
            x_sec_grid_v = np.concatenate((x_sec_grid_v, np.array(data[neut_type][int_type][energy_val]["ds/dy"])))
   
    
        xsec_interp = interpolate.LinearNDInterpolator((energy_grid_v, y_grid_v),
                                           x_sec_grid_v,
                                            fill_value=0)
    
        avg_y_index[neut_type][int_type] = avg_y_interp
        dsdy_index[neut_type][int_type] = xsec_interp

def XSectionReweights(frame):
    if (not frame.Has("MuonWeight")) and frame.Has("I3MCWeightDict"):
        print("Reweighting XC")
        energy = frame["I3MCWeightDict"]["PrimaryNeutrinoEnergy"]
        true_y = frame["I3MCWeightDict"]["BjorkenY"]
        InteractionType = frame["I3MCWeightDict"]["InteractionType"]
        InIceNeutrinoType = frame["I3MCWeightDict"]["InIceNeutrinoType"]

        if InteractionType == 3:
            a_minus = 1.0
            a_plus = 1.0

            return True

        if energy < 1e3:
            energy = 1e3
            
        a_minus, a_plus = get_weight_bounds(energy, true_y, InIceNeutrinoType, InteractionType)
        rw_vals = [a_minus, 1.0, a_plus]
        sigma_vals = [-1.0, 0.0, 1.0]
        poly,pcov= curve_fit(line_fit, sigma_vals[:],rw_vals[:],maxfev=2000) 

        frame['Inelasticity_param_a']=dataclasses.I3Double(poly[0])
        frame['Inelasticity_param_b']=dataclasses.I3Double(poly[1])       
tray.Add(XSectionReweights, "XSection Reweights")         
    
            

def AddPassingFractions(frame):
    #print("Adding first PF")
    Flav_dict={12:'conv nu_e',-12:'conv nu_ebar',14:'conv nu_mu',-14:'conv nu_mubar',16:'nu_tau',-16:'nu_tau'}
    energy_space= np.logspace(1,7,51)
    low_angles= np.linspace(0,0.5,51)
    high_angles=np.linspace(0.5,1,26)
    new_low_angles=np.delete(low_angles,0)
    angles_space=np.concatenate( (new_low_angles,high_angles) )
    depth_space = np.linspace(1.5,2.5,26)

    try:
        pdg_val=int(frame["I3MCWeightDict"]["PrimaryNeutrinoType"])
        if (abs(pdg_val) ==12 or abs(pdg_val) ==14):#primary is a neutrino
            zenith=frame["I3MCWeightDict"]["PrimaryNeutrinoZenith"]
            depth=frame['PrimaryDepthMC'].value/1000
            energy=frame['PrimaryEnergyMC'].value
            flav_val=Flav_dict[pdg_val]
            angle_val=np.around(angles_space[np.digitize(np.cos(zenith),angles_space)-1],decimals=2)
            depth_val=np.around(depth_space[np.digitize(depth,depth_space)-1],decimals=2)
            outfile_name = "PF_plight_combo_mymudet_MCPE3_energy_range_10GeV_1PeV_angle_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val,str(depth_val))
            outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/veto_array_"+outfile_name+".npy"
            x=np.load(outfile)
            PF=np.interp(energy,energy_space,x)
            # print('PF',PF)
            frame['AtmNeutrinoPassingFraction']=dataclasses.I3Double(PF)
        elif abs(pdg_val) == 16:
            frame['AtmNeutrinoPassingFraction']=dataclasses.I3Double(1)
    except Exception as e:
        frame['AtmNeutrinoPassingFraction']=dataclasses.I3Double(1)
        #print(frame['AtmNeutrinoPassingFraction'],e)
if not is_burn:
    print("Not burn - adding passing fractions")
    tray.AddModule(AddPassingFractions, "PassingFractions")

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return (idx-1)
    else:
        return idx

#def PaddedDepth(p):
#
#    for pad in np.linspace(0,500,100):
#        bound_2D,surface_det = get_surface_det(gcdFile=gcdFile,GCDPads=pad)
#        intersection=surface_det.intersection(p.pos, p.dir)#points of intersection
#        z_inter=p.pos.z-intersection.first*np.cos(p.dir.zenith)
#        if np.isnan(z_inter):
#            continue
#        depth=1948.07-z_inter
#
#        break
#    return(depth)


def AddNewPassingFractions(frame):
    # print('Searching for PAssingFractions')
    energy_space= np.logspace(1,7,51)
    angles_space=np.linspace(-1,1,10+1)[5:]
    # print(angles_space)
    low_depth_space = np.linspace(1.4,2.0,1+1)
    high_depth_space = np.linspace(2.1,2.5,1+1)
    depth_space=np.concatenate((low_depth_space,high_depth_space))

    if "I3MCWeightDict" not in frame:
        return
    pdg_val=int(frame["I3MCWeightDict"]["PrimaryNeutrinoType"])
    if (abs(pdg_val) ==12 or abs(pdg_val) ==14 or abs(pdg_val) ==16):#primary is a neutrino
        zenith=frame["I3MCWeightDict"]["PrimaryNeutrinoZenith"]
        # depth=frame['PrimaryDepthMC'].value/1000
        energy=frame['PrimaryEnergyMC'].value

        # x_vals=[-2,0,2]
        # x_vals=[-1,0,1]

        if np.cos(zenith)<0:
            # print('Upgoing Event')
            frame['AtmNeutrinoPassingFraction_New']=dataclasses.I3Double(1)
            frame['AtmNeutrinoPassingFraction_New_Pr']=dataclasses.I3Double(1)
            frame['AtmNeutrinoPassingFraction_New_Lower']=dataclasses.I3Double(1)
            frame['AtmNeutrinoPassingFraction_New_Pr_Lower']=dataclasses.I3Double(1)
            frame['AtmNeutrinoPassingFraction_New_Upper']=dataclasses.I3Double(1)
            frame['AtmNeutrinoPassingFraction_New_Pr_Upper']=dataclasses.I3Double(1)
            frame['AtmNeutrinoPassingFraction_New_IntermLower']=dataclasses.I3Double(1)
            frame['AtmNeutrinoPassingFraction_New_Pr_IntermLower']=dataclasses.I3Double(1)
            frame['AtmNeutrinoPassingFraction_New_IntermUpper']=dataclasses.I3Double(1)
            frame['AtmNeutrinoPassingFraction_New_Pr_IntermUpper']=dataclasses.I3Double(1)
            frame['AtmNeutrinoPassingFraction_New_a']=dataclasses.I3Double(0.5)
            frame['AtmNeutrinoPassingFraction_New_Pr_a']=dataclasses.I3Double(0.5)
            frame['AtmNeutrinoPassingFraction_New_b']=dataclasses.I3Double(0)
            frame['AtmNeutrinoPassingFraction_New_Pr_b']=dataclasses.I3Double(0)
            frame['AtmNeutrinoPassingFraction_New_c']=dataclasses.I3Double(1e-10)
            frame['AtmNeutrinoPassingFraction_New_Pr_c']=dataclasses.I3Double(1e-10)
            # x_vals=[-3,-1,0,3,10]
            # x_vals=np.array(x_vals)
            # x_shift=x_vals+5.1
            # print(erfc_fit(x_shift,1,0,1e-10))
            # print()

            return
        if np.isnan(frame['PrimaryDepthMC'].value):#Skimming events get assigned to bin closest to padded depth
            del frame['PrimaryDepthMC']
            if 'I3MCTree_preMuonProp' in frame:
                mctree = frame['I3MCTree_preMuonProp']
                neutrino = None
                for p in mctree:
                    depth = mctree.depth(p)
                    if (depth == 0):
                        if neutrino is None and len(mctree.get_daughters(p))>0:
                            neutrino = p
                            return False
                            #frame["PrimaryDepthMC"]=dataclasses.I3Double(PaddedDepth(p))
                            #break
        depth=frame['PrimaryDepthMC'].value/1000
        angle_val=np.around(angles_space[np.digitize(np.cos(zenith),angles_space)-1],decimals=2)
        depth_val=np.around(depth_space[np.digitize(depth,depth_space)-1],decimals=2)


        Flav_dict_conv={12:'conv nu_e',-12:'conv nu_ebar',14:'conv nu_mu',-14:'conv nu_mubar',16:'conv nu_e',-16:'conv nu_ebar'}#same PFs for NuE and NuTau
        flav_val_conv=Flav_dict_conv[pdg_val]
        # conv_outfile_name = "PF_baseline_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_conv,str(depth_val))
        # conv_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE/veto_array_"+conv_outfile_name+".npy" 
        # plus_outfile_name = "PF_plussshift_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_conv,str(depth_val))
        # plus_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE/veto_array_"+plus_outfile_name+".npy" 
        # minus_outfile_name = "PF_minussshift_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_conv,str(depth_val))
        # minus_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE/veto_array_"+minus_outfile_name+".npy" 
        conv_outfile_name = "PF_baseline_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_conv,str(depth_val))
        conv_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE_v2/veto_array_"+conv_outfile_name+".npy"
        plus_outfile_name = "PF_plussshift_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_conv,str(depth_val))
        plus_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE_v2/veto_array_"+plus_outfile_name+".npy"
        intermplus_outfile_name = "PF_intermplussshift_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_conv,str(depth_val))
        intermplus_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE_v2/veto_array_"+intermplus_outfile_name+".npy"

        minus_outfile_name = "PF_minussshift_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_conv,str(depth_val))
        minus_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE_v2/veto_array_"+minus_outfile_name+".npy"
        intermminus_outfile_name = "PF_intermminussshift_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_conv,str(depth_val))
        intermminus_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE_v2/veto_array_"+intermminus_outfile_name+".npy"
        if os.path.exists(conv_outfile):#exact PF exists
            # print('Conv PF exists')
            x=np.load(conv_outfile)
            PF=np.interp(energy,energy_space,x)
            # print('Conv PF',PF)
            frame['AtmNeutrinoPassingFraction_New']=dataclasses.I3Double(PF)
            x=np.load(plus_outfile)
            PF_upper=np.interp(energy,energy_space,x)
            # print('Plus Conv PF',PF_upper)
            frame['AtmNeutrinoPassingFraction_New_Upper']=dataclasses.I3Double(PF_upper)
            x=np.load(minus_outfile)
            PF_lower=np.interp(energy,energy_space,x)
            # print('Minus Conv PF',PF_lower)
            frame['AtmNeutrinoPassingFraction_New_Lower']=dataclasses.I3Double(PF_lower)
            x=np.load(intermplus_outfile)
            PF_intermupper=np.interp(energy,energy_space,x)
            # print('IntermPlus Conv PF',PF_intermupper)
            frame['AtmNeutrinoPassingFraction_New_IntermUpper']=dataclasses.I3Double(PF_intermupper)
            x=np.load(intermminus_outfile)
            PF_intermlower=np.interp(energy,energy_space,x)
            # print('IntermMinus Conv PF',PF_intermlower)
            frame['AtmNeutrinoPassingFraction_New_IntermLower']=dataclasses.I3Double(PF_intermlower)
            # poly = np.polyfit(x_vals, [PF_lower,PF_intermlower,PF,PF_intermupper,PF_upper], deg=2)
            # print(poly)
            try:
                x_vals=[-3,-1,0,3,10]
                x_vals=np.array(x_vals)
                x_shift=x_vals+5.1
                y_vals=[PF_lower,PF_intermlower,PF,PF_intermupper,PF_upper]
                y_vals=np.array(y_vals)
                poly,pcov= curve_fit(erfc_fit,x_shift[:],y_vals[:],p0=[1,5,1],maxfev=2000,bounds =(0, [np.inf, np.inf, np.inf]))

                # print(poly)
                # print(erfc_fit(x_shift,*poly))
                # print(y_vals)
                # print()
            except Exception as e:
                print(e,frame['I3EventHeader'])
                x_vals=[-1,0,3]
                x_vals=np.array(x_vals)
                x_shift=x_vals+5.1
                y_vals=[PF_intermlower,PF,PF_intermupper]
                y_vals=np.array(y_vals)
                poly,pcov= curve_fit(erfc_fit,x_shift[:],y_vals[:],p0=[1,5,1],maxfev=2000,bounds =(0, [np.inf, np.inf, np.inf]))

            frame['AtmNeutrinoPassingFraction_New_a']=dataclasses.I3Double(poly[0])
            frame['AtmNeutrinoPassingFraction_New_b']=dataclasses.I3Double(poly[1])
            frame['AtmNeutrinoPassingFraction_New_c']=dataclasses.I3Double(poly[2])
        else:
            print('Conv PF does not exist!',frame['I3EventHeader'])
            print('Zenith, Depth',np.cos(zenith),depth)
            print('Binned Zenith, Depth',angle_val,depth_val)

        Flav_dict_pr={12:'pr nu_e',-12:'pr nu_ebar',14:'pr nu_mu',-14:'pr nu_mubar',16:'pr nu_e',-16:'pr nu_ebar'}
        flav_val_pr=Flav_dict_pr[pdg_val]
        # pr_outfile_name = "PF_baseline_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_pr,str(depth_val))
        # pr_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE/veto_array_"+pr_outfile_name+".npy" 
        # plus_outfile_name = "PF_plussshift_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_pr,str(depth_val))
        # plus_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE/veto_array_"+plus_outfile_name+".npy" 
        # minus_outfile_name = "PF_minussshift_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_pr,str(depth_val))
        # minus_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE/veto_array_"+minus_outfile_name+".npy" 
        pr_outfile_name = "PF_baseline_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_pr,str(depth_val))
        pr_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE_v2/veto_array_"+pr_outfile_name+".npy"
        plus_outfile_name = "PF_plussshift_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_pr,str(depth_val))
        plus_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE_v2/veto_array_"+plus_outfile_name+".npy"
        minus_outfile_name = "PF_minussshift_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_pr,str(depth_val))
        minus_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE_v2/veto_array_"+minus_outfile_name+".npy"
        intermplus_outfile_name = "PF_intermplussshift_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_pr,str(depth_val))
        intermplus_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE_v2/veto_array_"+intermplus_outfile_name+".npy"
        intermminus_outfile_name = "PF_intermminussshift_injectionmudet_%s_neut_type_%s_at_depth_%skm"%(str(angle_val),flav_val_pr,str(depth_val))
        intermminus_outfile = "/data/user/vbasu/CONDOR_output/MESE_L5_PropNu_PF_new_plight/LowE_v2/veto_array_"+intermminus_outfile_name+".npy"

        if os.path.exists(pr_outfile):#exact PF exists
            # print('PR PF exists')
            x=np.load(pr_outfile)
            PF=np.interp(energy,energy_space,x)
            # print('Prompt PF',PF)
            frame['AtmNeutrinoPassingFraction_New_Pr']=dataclasses.I3Double(PF)
            x=np.load(plus_outfile)
            PF_upper=np.interp(energy,energy_space,x)
            # print('Prompt Plus PF',PF_upper)
            frame['AtmNeutrinoPassingFraction_New_Pr_Upper']=dataclasses.I3Double(PF_upper)
            x=np.load(minus_outfile)
            PF_lower=np.interp(energy,energy_space,x)
            # print('Prompt Minus PF',PF_lower)
            frame['AtmNeutrinoPassingFraction_New_Pr_Lower']=dataclasses.I3Double(PF_lower)
            x=np.load(intermplus_outfile)
            PF_intermupper=np.interp(energy,energy_space,x)
            # print('IntermPlus Conv PF',PF_intermupper)
            frame['AtmNeutrinoPassingFraction_New_Pr_IntermUpper']=dataclasses.I3Double(PF_intermupper)
            x=np.load(intermminus_outfile)
            PF_intermlower=np.interp(energy,energy_space,x)
            # print('Minus Conv PF',PF_intermlower)
            frame['AtmNeutrinoPassingFraction_New_Pr_IntermLower']=dataclasses.I3Double(PF_intermlower)
            # poly = np.polyfit(x_vals, [PF_lower,PF_intermlower,PF,PF_intermupper,PF_upper], deg=2)
            # print(poly)


            try:
                x_vals=[-3,-1,0,3,10]
                x_vals=np.array(x_vals)
                x_shift=x_vals+5.1
                y_vals=[PF_lower,PF_intermlower,PF,PF_intermupper,PF_upper]
                y_vals=np.array(y_vals)
                poly,pcov= curve_fit(erfc_fit,x_shift[:],y_vals[:],p0=[1,5,1],maxfev=2000,bounds =(0, [np.inf, np.inf, np.inf]))
            except Exception as e:
                #print(e,frame['I3EventHeader'])
                x_vals=[-1,0,3]
                x_vals=np.array(x_vals)
                x_shift=x_vals+5.1
                y_vals=[PF_intermlower,PF,PF_intermupper]
                y_vals=np.array(y_vals)
                poly,pcov= curve_fit(erfc_fit,x_shift[:],y_vals[:],p0=[1,5,1],maxfev=2000,bounds =(0, [np.inf, np.inf, np.inf]))

            frame['AtmNeutrinoPassingFraction_New_Pr_a']=dataclasses.I3Double(poly[0])
            frame['AtmNeutrinoPassingFraction_New_Pr_b']=dataclasses.I3Double(poly[1])
            frame['AtmNeutrinoPassingFraction_New_Pr_c']=dataclasses.I3Double(poly[2])
        else:
            print('PR PF does not exist!',frame['I3EventHeader'])
            print('Zenith, Depth',np.cos(zenith),depth)
            print('Binned Zenith, Depth',angle_val,depth_val)

tray.AddModule(AddNewPassingFractions, "NewPassingFractions")

def AddNewMuonWeights(frame, number_of_files):
    #print("Adding NewWeights")
    if frame.Has("MuonWeight"):
        MuonWeights = frame["MuonWeight"].value
        New_weights = MuonWeights/number_of_files
        frame["MuonWeight_corrected2"] = dataclasses.I3Double(New_weights)
    else:
        return True
tray.AddModule(AddNewMuonWeights, "Add_New_Muon_Weights", number_of_files = number_of_files)

def RemoveBadYEff(frame):
    if frame.Has("YEff_milli_inDet") and frame.Has("IsCascade_dnn"):
        yeff = frame["YEff_milli_inDet"].value
        #print(yeff)
        if frame["IsCascade_dnn"].value == 1:
            #print("Cascade")
            return True
        else:
            #print("Track")
            return (yeff >= 0.01)
#tray.AddModule(RemoveBadYEff, "Remove Bad YEff")    



def RecoVariablesForNNMFit(frame):
    if not frame.Has("YEff_milli"):
        energy = frame["RecoEnergy"]
        if energy < 1*10**(4.5):
            if energy < 1*10**(4):
                frame["YEff_milli"] = frame["yeff_5m"]
            else:
                frame["YEff_milli"] = frame["yeff_7point5m"]
        else:
            if energy < 1*10**(5):
                frame["YEff_milli"] = frame["yeff_10m"]
            else:
                frame["YEff_milli"] = frame["yeff_12point5m"]

#tray.AddModule(RecoVariablesForNNMFit, "RecoVarGetter")

def glashow_correction(frame):
    if 'I3MCWeightDict' in frame:
        from scipy.interpolate import interp1d
        nutype=int(frame["I3MCWeightDict"]["PrimaryNeutrinoType"])
        inter_type=int(frame['I3MCWeightDict']['InteractionType'])
        en = float(frame['I3MCWeightDict']['PrimaryNeutrinoEnergy'])
        if (abs(nutype)==12 and inter_type==3.0 and en>4e6):
            old_spline=pd.read_csv('/home/abalagopalv/diffuse/TauStudies/Glashow_old.csv',header=None)
            new_spline=pd.read_csv('/home/abalagopalv/diffuse/TauStudies/Glashow_new.csv',header=None)

            x = old_spline[0]
            y = old_spline[1]

            xn = new_spline[0]
            yn = new_spline[1]
            f1 = interp1d(x, y, kind='cubic')
            f2 = interp1d(xn, yn, kind='cubic')
            if en<9.9e6:
                
                num = f2(en/1e6)
                denom = f1(en/1e6)
                ratio = num/denom
                frame['TotalWeight'] = dataclasses.I3Double(frame['I3MCWeightDict']['TotalWeight']*ratio)
            elif en>=9.9e6:
                num = f2(9.89)
                denom = f1(9.89)
                ratio = num/denom
                frame['TotalWeight'] = dataclasses.I3Double(frame['I3MCWeightDict']['TotalWeight']*ratio)
        else:
            frame['TotalWeight'] = dataclasses.I3Double(frame['I3MCWeightDict']['TotalWeight'])
            
tray.Add(glashow_correction)

pol_hadr_0 = '/home/abalagopalv/diffuse/TauStudies/tau_polarization/Hadron_polarization_0.csv'
pol_hadr_minus1='/home/abalagopalv/diffuse/TauStudies/tau_polarization/Hadron_polarization_-1.csv'
pol_lep_0 = '/home/abalagopalv/diffuse/TauStudies/tau_polarization/Lepton_polarization_0.csv'
pol_lep_minus1='/home/abalagopalv/diffuse/TauStudies/tau_polarization/Lepton_polarization_-1.csv'


def tau_polarization(frame):
    if not frame.Has("I3MCWeightDict"):
        return True
    leptons = [dataclasses.I3Particle.MuPlus, dataclasses.I3Particle.MuMinus, dataclasses.I3Particle.EMinus, dataclasses.I3Particle.EPlus]
    neutrinos = [dataclasses.I3Particle.NuE, dataclasses.I3Particle.NuEBar,
                     dataclasses.I3Particle.NuMu, dataclasses.I3Particle.NuMuBar,
                     dataclasses.I3Particle.NuTau, dataclasses.I3Particle.NuTauBar]
    from scipy.interpolate import interp1d
    nutype=frame["I3MCWeightDict"]["PrimaryNeutrinoType"]
    inter_type=frame['I3MCWeightDict']['InteractionType']
    had_energy = []
    lep_energy = []
    neut_energy = []
    print(frame['I3EventHeader'].run_id,frame['I3EventHeader'].event_id, inter_type, nutype)
    if (abs(nutype)==16 and inter_type==1.0):
        MCTreeName = 'I3MCTree'
        if frame.Has(MCTreeName):
            prims, _ = FindTrueParent(frame["I3MCTree"])
            for p in [prims]:
            #for p in frame[MCTreeName].get_primaries():
                if (p.type != dataclasses.I3Particle.NuTau and p.type != dataclasses.I3Particle.NuTauBar):
                    continue
                for c in frame[MCTreeName].children(p):
                    #print("Children type: ", c.type)
                    if c.type == dataclasses.I3Particle.TauMinus or c.type == dataclasses.I3Particle.TauPlus:
                        E_Tau = c.energy

                        for d in frame[MCTreeName].get_daughters(c):
                            if d.type == dataclasses.I3Particle.NuTau or d.type == dataclasses.I3Particle.NuTauBar:
                                time_of_int = d.time
                                break
                            else:
                                time_of_int = 0.   #to account for Tau decays outside of detector

                        for d in frame[MCTreeName].get_daughters(c):
                            if d.time == time_of_int and d.type in neutrinos:
                                neut_energy.append(d.energy)            

                            if d.time == time_of_int and d.type not in neutrinos:
                                    if d.type not in leptons:

                                        had_energy.append(d.energy)

                                    elif d.type in leptons:
                                        lep_energy.append(d.energy)

                        y_lep = sum(lep_energy)/E_Tau
                        y_had = sum(had_energy)/E_Tau
                        y_neut = 1-(sum(neut_energy)/E_Tau)
                        #print(y_lep, y_had, y_neut)
    
    if sum(lep_energy)!=0:
        #print("Lepton energy non-zero %s" %y_lep)
        old_spline=pd.read_csv(pol_lep_0,header=None)
        new_spline=pd.read_csv(pol_lep_minus1,header=None)
        x = old_spline[0]
        y = old_spline[1]
        xn = new_spline[0]
        yn = new_spline[1]
        f1 = interp1d(x, y, kind='cubic')
        f2 = interp1d(xn, yn, kind='cubic')

        num = f2(y_lep)
        denom = f1(y_lep)
        ratio = num/denom
        frame['TotalWeightPol'] = dataclasses.I3Double(frame['TotalWeight'].value*ratio)
        frame["y_neut"] = dataclasses.I3Double(y_neut)
        frame['y_lep'] = dataclasses.I3Double(y_lep)
    elif sum(had_energy)!=0:
        #print("Had energy non-zero %s" %y_had)
        old_spline=pd.read_csv(pol_hadr_0,header=None)
        new_spline=pd.read_csv(pol_hadr_minus1,header=None)
        x = old_spline[0]
        y = old_spline[1]
        xn = new_spline[0]
        yn = new_spline[1]
        f1 = interp1d(x, y, kind='cubic')
        f2 = interp1d(xn, yn, kind='cubic')

        num = f2(y_had)
        denom = f1(y_had)
        ratio = num/denom
        frame['TotalWeightPol'] = dataclasses.I3Double(frame['TotalWeight'].value*ratio)
        frame["y_neut"] = dataclasses.I3Double(y_neut)
        frame['y_had']= dataclasses.I3Double(y_had)


    if not frame.Has('TotalWeightPol'):
        frame['TotalWeightPol'] = dataclasses.I3Double(frame['TotalWeight'].value)

tray.Add(tau_polarization)

import matplotlib.path as mpltPath

def select(geometry):
        strings = collections.defaultdict(list)
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
    surface_det = MuonGun.ExtrudedPolygon.from_file(MuonGunGCD, padding=500)##Build Polygon from I3Geometry (convex hull)
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
    
def yeff_onepercent(frame):
    if frame["IsCascade_dnn"] == 1:
        frame["YEff_onepercent"] = dataclasses.I3Double(0.5)
        return True
    #print("Millipede yeff check")
    if frame.Has("Millipede_SplineMPE_TWTS"):
        cascades = frame["Millipede_SplineMPE_TWTS"]

        #global E_total_from_losses

        non_zero_cascades = 0
        E_total_from_losses = 0
        E_Vector = []
        E_inDet = frame["RecoEnergy"].value
        if E_inDet == 0:
            print(frame["I3EventHeader"])
            frame["YEff_onepercent"] = dataclasses.I3Double(float("nan"))
            return True


        for cascade in cascades:
            if boundary_check(cascade): # and cascade.time >= frame["MillipedeFirstLoss_TWTS"].time:
                ####    Just summing the energy for E_total_from_losses gets you the total along the entire set of cascades
                ####    For just the energy inside the detector use frame["MillipedeDepositedEnergy_TWTS"]
                E_total_from_losses += cascade.energy   ##HERE 
                E_Vector.append(float(cascade.energy))  ## AND HERE
        #PRint(len(E_Vector))
        E_Vector = np.array(E_Vector)
        E_Vector_no_zero = E_Vector[E_Vector > 0.01*E_inDet]
        if len(E_Vector_no_zero) == 0:
            print("No cascade deposits above 1% Total")
            return False


        #print("Is first deposit same as Vedant says? %s" %(E_Vector_no_zero[0] == frame["MillipedeFirstLoss_TWTS"].energy))
        #print("Has FirstLoss? ", frame.Has("MillipedeFirstLoss_TWTS"))
        #print(len(E_Vector_no_zero))
        E_inDet = frame["MillipedeDepositedEnergy_TWTS"].value
        if len(E_Vector_no_zero) == 0:
            print("No non-zero reco particle")
            yeff_2point5m = 0
            yeff_5m = 0
            yeff_7point5m = 0
            yeff_10m = 0
            yeff_12point5m = 0

        else:
            if not (E_Vector_no_zero[0] == frame["MillipedeFirstLoss_TWTS"].energy):
                print(E_Vector_no_zero[0])
                print(frame["MillipedeFirstLoss_TWTS"].energy)

            yeff_2point5m = E_Vector_no_zero[0]/E_inDet
            #yeff_2point5m_det = E_Vector_no_zero[0]/E_inDet
            ##print("Minimum non-zero at %s, length is %s" %(np.min(np.where(E_Vector==E_Vector_no_zero[0])), len(E_Vector)))
            if np.min(np.where(E_Vector==E_Vector_no_zero[0])) +1 >= len(E_Vector):
                yeff_5m = yeff_2point5m
            else:
                yeff_5m = (E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))]+E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+1])/E_inDet
            if np.min(np.where(E_Vector==E_Vector_no_zero[0])) +2 >= len(E_Vector):
                yeff_7point5m  = yeff_5m
            else:
                yeff_7point5m = (E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+1] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+2])/E_inDet
            if np.min(np.where(E_Vector==E_Vector_no_zero[0]))+3 >= len(E_Vector):
                yeff_10m = yeff_7point5m
            else:
                yeff_10m = (E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+1] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+2] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+3])/E_inDet
            if np.min(np.where(E_Vector==E_Vector_no_zero[0]))+4 >= len(E_Vector):
                yeff_12point5m = yeff_10m
            else:
                yeff_12point5m = (E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+1] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+2] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+3] + E_Vector[np.min(np.where(E_Vector==E_Vector_no_zero[0]))+4])/E_inDet

    else:
        yeff_dummy = float("nan")

        yeff_2point5m = yeff_dummy
        yeff_5m = yeff_dummy
        yeff_7point5m = yeff_dummy
        yeff_10m = yeff_dummy
        yeff_12point5m = yeff_dummy
    energy = frame["RecoEnergy"]
    if energy < 1*10**(4.5):
        if energy < 1*10**(4):
            frame["YEff_onepercent"] = frame["yeff_5m"]
        else:
            frame["YEff_onepercent"] = frame["yeff_7point5m"]
    else:
        if energy < 1*10**(5):
            frame["YEff_onepercent"] = frame["yeff_10m"]
        else:
            frame["YEff_onepercent"] = frame["yeff_12point5m"]

tray.AddModule(yeff_onepercent, "Yeff Millipede ver")




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
                'MuonWeight_corrected', "TotalWeight",
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
                'RecoEnergy','RecoParticle','RecoDirection','SnowstormParameterDict','AtmNeutrinoPassingFraction',
                "NEvents", "RandomForestEnergyEstimates","RFYeff", "YEff_milli", "YEff_milli_inDet", "YEff_TrueDep", 
                "yeff_2point5m", "yeff_5m", "yeff_7point5m", "yeff_10m", "yeff_12point5m",
                "AtmNeutrinoPassingFraction_New", "AtmNeutrinoPassingFraction_New_Upper", "AtmNeutrinoPassingFraction_New_Lower", "AtmNeutrinoPassingFraction_New_IntermUpper", 
                "AtmNeutrinoPassingFraction_New_IntermLower", "AtmNeutrinoPassingFraction_New_a", "AtmNeutrinoPassingFraction_New_b", "AtmNeutrinoPassingFraction_New_c",
                 "AtmNeutrinoPassingFraction_New_Pr", "AtmNeutrinoPassingFraction_New_Pr_Upper", "AtmNeutrinoPassingFraction_New_Pr_Lower", 
                 "AtmNeutrinoPassingFraction_New_Pr_IntermUpper", "AtmNeutrinoPassingFraction_New_Pr_IntermLower", "AtmNeutrinoPassingFraction_New_Pr_a", 
                "AtmNeutrinoPassingFraction_New_Pr_b", "AtmNeutrinoPassingFraction_New_Pr_c",
                "MuonWeight_corrected2", "Inelasticity_param_a", "Inelasticity_param_b", "y_had", "y_lep", "y_neut", "TotalWeightPol", "y_eff_true", "E_close", "E_far", "YEff_onepercent"]

if int(filenum) in [22010, 22011, 22012, 22013, 22014, 22015, 22016, 22017, 22018]:
    outfile = "/data/user/rclark/TauSimProject/InelasicityHDF/L5_NewIce_%s_%sfiles_NewPolarisationNewInel_2XC_AboveOnePercent" %(filenum, number_of_files)
elif int(filenum) in [21315, 21316, 21317, 21318, 21319]:
    outfile = "/data/user/rclark/TauSimProject/InelasicityHDF/L5_NewIce_%s_%sfiles_AboveOnePercent" %(filenum, number_of_files)
elif int(filenum) in [2016, 2017]:
    outfile = "/data/user/rclark/TauSimProject/InelasicityHDF/L5_NewIce_Burn_%s_%sfiles_AboveOnePercent" %(filenum, number_of_files)
elif int(filenum) in [2011, 2012, 2013, 2014, 2015, 2018, 2019, 2020, 2021]:
    outfile = "/data/user/rclark/TauSimProject/InelasicityHDF/L5_NewIce_Stage1_%s_%sfiles_RemoveBadYEff" %(filenum, number_of_files)

#outfile = "/data/user/rclark/TauSimProject/InelasicityHDF/L5_NewIce_%s_%sfiles" %(filenum, number_of_files)
#outfile = args.outfile+"/L5_NewIce_Stage1_%s_%sfiles" %(filenum, number_of_files)
print(outfile)

tray.Add(
    hdfwriter.I3HDFWriter,
    SubEventStreams=["InIceSplit", "topological_split"],
    keys = table_keys,
    output=outfile+".hdf5",
    )
print(outfile+".hdf5")
tray.AddModule("TrashCan", "thecan")
tray.Execute()
tray.Finish()


stop_time = time.asctime()
print("Stopping: ", stop_time)
