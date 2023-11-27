import os,pickle,gzip,urllib,sys, re

from icecube import photonics_service

def save_obj(obj, name ):
    f = gzip.open(name,"wb")
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    
def load_obj(name):
    f = gzip.open(name,"rb")
    return pickle.load(f)
    
def return_fitnames():
    fitnames=["SPEFit2","MPEFit","PoleMuonLlhFit",
              "OnlineL2_PoleL2MPEFit","SPEFit12EHE","SPEFitSingle",
              "SPEFitSingleEHE","MPEFit4","MPEFitHighNoise",
              "MuEXAngular4","SPEFit8","SplineMPEMuEXDifferential",
              "SplineMPE_default","SplineMPE_recommended","SplineMPESPEFit2_Free_1_10"]
    return fitnames
    
def add_holes_to_check_to_fitnames(fitnames,fn="Holes_To_Check.pklz"):
    fp=sys.path[0]+"/"+fn
    if not os.path.isfile(fp):
        urllib.urlretrieve ("http://icecube.wisc.edu/~kjero/"+fn,fp)
    if not os.path.isfile(fn):
        urllib.urlretrieve ("http://icecube.wisc.edu/~kjero/"+fn,fn)
    holes_to_check = load_obj(fp)                             
    for htc_idx,htc in enumerate(holes_to_check):
        fitnames.append("VetoFit"+str(htc_idx))
    return fitnames

def add_vseed_to_check_to_fitnames(fitnames):
    vseed_fitnames=[]
    for seed_idx in range(120*6*2):
        vseed_fitnames.append("seed"+str(seed_idx))
    fitnames.extend(vseed_fitnames)
    return fitnames

def return_edge_string_list_and_cap_DOMs_list():
    flat_edge_strs=set([1,2,3,4,5,6,13,21,30,40,50,59,67,74,73,72,78,77,76,75,68,60,51,41,31,22,14,7])
    top_DOMs=set([1,2,3,4,5])
    return [flat_edge_strs,top_DOMs]

def return_photonics_service(service_type="inf_muon"):
    table_base=""
    if os.path.isfile(os.path.expandvars("$I3_DATA/photon-tables/splines/ems_mie_z20_a10.%s.fits") % "abs"):
        table_base = os.path.expandvars("$I3_DATA/photon-tables/splines/ems_mie_z20_a10.%s.fits")
    elif os.path.isfile("splines/ems_mie_z20_a10.%s.fits" % "abs"):
        table_base = os.path.expandvars("splines/ems_mie_z20_a10.%s.fits")
    elif os.path.isfile("/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/ems_mie_z20_a10.%s.fits" % "abs"):
        table_base = os.path.expandvars("/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/ems_mie_z20_a10.%s.fits")
    elif os.path.isfile("/home/icecube/i3/data/generalized_starting_events/splines/ems_mie_z20_a10.%s.fits" % "abs"):
        table_base = os.path.expandvars("/home/icecube/i3/data/generalized_starting_events/splines/ems_mie_z20_a10.%s.fits")
    else:
        print ("You don't have splines anywhere I can find. This will eventually raise an error, for now it semi-silently dies")
    if service_type=="cscd":
        cascade_service = photonics_service.I3PhotoSplineService(
            amplitudetable=table_base % "abs", 
            timingtable=table_base % "prob", 
            timingSigma=0,
            maxRadius    = 600.0)
        return cascade_service
    elif service_type=="seg_muon":
        seg_muon_service = photonics_service.I3PhotoSplineService(
                           amplitudetable = os.path.join( os.path.expandvars("$I3_DATA/photon-tables/splines/") ,"ZeroLengthMieMuons_250_z20_a10.abs.fits"),  ## Amplitude tables 
                           timingtable = os.path.join( os.path.expandvars("$I3_DATA/photon-tables/splines/") ,"ZeroLengthMieMuons_250_z20_a10.prob.fits"),    ## Timing tables
                           timingSigma  = 0.0,
                           maxRadius    = 600.0)
        return seg_muon_service
    elif service_type=="inf_muon":
        inf_muon_service = photonics_service.I3PhotoSplineService(
                           amplitudetable = os.path.join( os.path.expandvars("$I3_DATA/photon-tables/splines/") ,"InfBareMu_mie_abs_z20a10.fits"),  ## Amplitude tables 
                           timingtable = os.path.join( os.path.expandvars("$I3_DATA/photon-tables/splines/") ,"InfBareMu_mie_prob_z20a10.fits"),    ## Timing tables
                           timingSigma  = 0.0,
                           maxRadius    = 600.0) 
        return inf_muon_service
    else:
        print ("You didn't give me a spline service type I recognize. This will eventually raise an error, for now it semi-silently dies")

def setup_fnames(use_grid, outfilebase, infiles_list):
    infiles =[]
    grid = 'gsiftp://gridftp.icecube.wisc.edu'
    my_grid = 'gsiftp://gridftp.icecube.wisc.edu/data/ana/Muon/ESTES/ESTES_2019/ESTES_data/'
    temp_outfile = outfilebase.lower()
    if("burn" in temp_outfile):    my_grid = my_grid + "burn_sample/"
    elif("all_data" in temp_outfile):my_grid = my_grid + "all_data/"
    elif("test" in temp_outfile): my_grid = my_grid + "test/"
    elif("manuel" in temp_outfile): my_grid = my_grid + "muongun_manuel/"
    elif("corsika_nu_10661" in temp_outfile): my_grid = my_grid + "corsika-nu/"
    elif("corsika_20904" in temp_outfile): my_grid = my_grid + "corsika/"
    elif("corsika" in temp_outfile): my_grid = my_grid + "corsika_mixed/"
    elif("21217" in temp_outfile):   my_grid = my_grid + "nugen_numu_21217/"
    elif("21218" in temp_outfile):   my_grid = my_grid + "nugen_nue_21218/"
    elif("21219" in temp_outfile):   my_grid = my_grid + "nugen_nutau_21219/"
    elif("21220" in temp_outfile):   my_grid = my_grid + "nugen_numu_21220/"
    elif("21221" in temp_outfile):   my_grid = my_grid + "nugen_nutau_21221/"
    elif("21633" in temp_outfile):   my_grid = my_grid + "nugen_numu_21633/"
    elif("21634" in temp_outfile):   my_grid = my_grid + "nugen_numu_21634/"
    elif("21635" in temp_outfile):   my_grid = my_grid + "nugen_numu_21635/"
    elif("21792" in temp_outfile):   my_grid = my_grid + "nugen_nutau_21792/"
    elif("21793" in temp_outfile):   my_grid = my_grid + "nugen_nutau_21793/"
    elif("21794" in temp_outfile):   my_grid = my_grid + "nugen_nutau_21794/"
    elif("21795" in temp_outfile):   my_grid = my_grid + "nugen_nue_21795/"
    elif("21796" in temp_outfile):   my_grid = my_grid + "nugen_nue_21796/"
    elif("21797" in temp_outfile):   my_grid = my_grid + "nugen_nue_21797/"
    elif("21124" in temp_outfile):   my_grid = my_grid + "nugen_numu_21124/"
    elif("21002" in temp_outfile):   my_grid = my_grid + "nugen_numu_21002/"
    elif( ("21813" in temp_outfile) or ("21814" in temp_outfile) or ("21938" in temp_outfile) or ("21867" in temp_outfile) or ("21868" in temp_outfile) or ("21939" in temp_outfile) or ("21870" in temp_outfile) or ("21871" in temp_outfile) or ("21940" in temp_outfile)):
        temp_gridname  = (temp_outfile).replace("nugen_numu_21813_","").replace("nugen_numu_21814_","").replace("nugen_numu_21938_","").replace("nugen_nutau_21867_","").replace("nugen_nutau_21868_","").replace("nugen_nutau_21939_","").replace("nugen_nue_21870_","").replace("nugen_nue_21871_","").replace("nugen_nue_21940_","")
        if("p0" in temp_outfile):
            temp_gridname = temp_gridname.split("_")[0]+"_"+temp_gridname.split("_")[1]
        else:
            temp_gridname =temp_gridname.split("_")[0]
#        if("scat" in temp_outfile) or ("abs" in temp_outfile):
#            temp_my_grid = temp_gridname.replace("scat","s").replace("abs","a")
#        elif("domeff" in temp_outfile):
#            temp_my_grid = "%s"%(temp_gridname.replace("domeff","domeff="))
#        if("p0" in temp_outfile):
#            temp_my_grid = "%s"%temp_gridname
        temp_my_grid=temp_gridname
        my_grid = my_grid+temp_outfile.split("_")[0]+"_"+temp_outfile.split("_")[1]+"_"+temp_outfile.split("_")[2]+"_"+temp_my_grid+"/"
    if(use_grid):
        for nfile in infiles_list:
            if("cvmfs" in nfile):
                infiles.append(nfile)
            elif(("data/ana" in nfile) or("data/sim" in nfile) or ("data/exp" in nfile)):
                infiles.append(grid+nfile)
            else:
                infiles.append(my_grid+nfile)
    else:
        infiles = infiles_list
    
    return my_grid, infiles
