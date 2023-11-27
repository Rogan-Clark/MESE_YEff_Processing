from icecube import dataclasses,icetray,dataio,phys_services,VHESelfVeto,StartingTrackVeto, hdfwriter, recclasses, simclasses
import collections
import numpy as np
import matplotlib.path as mpltPath


def select_parent(geometry):
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

def boundaries_parent(geometry):
        top_layer=90.*icetray.I3Units.m,
        dust_layer=(-220.*icetray.I3Units.m,-100.*icetray.I3Units.m)
        strings = select_parent(geometry)
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

def get_surface_det_parent(gcdFile=None):

    from icecube import MuonGun
    gcdFile=gcdFile
    bound_2D=[]
    MuonGunGCD='/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz'
    surface_det = MuonGun.ExtrudedPolygon.from_file(MuonGunGCD, padding=500)##Build Polygon from I3Geometry (convex hull)
    f = dataio.I3File(MuonGunGCD)
    omgeo = f.pop_frame(icetray.I3Frame.Geometry)['I3Geometry'].omgeo
    surface_det_x,surface_det_y=boundaries_parent(omgeo)#getting this from omgeo gives concave hull instead of convex hull
    x=[(surface_det_x[i],surface_det_y[i])for i in range(len(surface_det_x))]###getting only x and y
    bound_2D=mpltPath.Path(x)#Projection of detector on x,y plane

    return bound_2D, surface_det

def boundary_check_parent(particle1,gcdFile=None):
    ####checks if particle is inside the detector###
    gcdFile=gcdFile
    # bound_2D,surface_det = get_surface_det(gcdFile=gcdFile)
    inlimit = False
    if ((particle1.pos.z <=max(surface_det.z)) and (particle1.pos.z>=min(surface_det.z))):
        if bound_2D.contains_points([(particle1.pos.x, particle1.pos.y)]):
            inlimit=True

    return inlimit


###Function, given MCTree, to find the parent neutrino in ice

def FindTrueParent(tree):
    ###wd = frame['I3MCWeightDict']

    ## If a tau or a Glashow resonance, we need to find the decay products
    global surface_det
    global bound_2D
    bound_2D,surface_det = get_surface_det_parent(gcdFile='/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz')
  
    #tree_preprop = frame["I3MCTree"]
    tree_preprop = tree
    printval=False
    for p in tree_preprop.get_primaries():
        if not abs(p.type) in [12,14,16]:
            continue

        c = tree_preprop.first_child(p)
        while (not boundary_check_parent(c)) or (not abs(c.type) in [11,12,13,14,15,16]):
            if tree_preprop.number_of_children(c) == 0:
                try:
                    c = tree_preprop.parent(c)
                except:
                    #print(c.type.name, c.minor_id, "No children or parents")
                    #print(frame["I3EventHeader"])
                    return p, p
                GC = tree_preprop.first_child(c)
                tree_preprop.erase(GC)
            else:
                c = tree_preprop.first_child(c)
        nu2 = tree_preprop.parent(c)

        while nu2.type not in [-12, -14, -16, 12, 14, 16]:
            nu2 = tree_preprop.parent(nu2)
        if nu2.is_neutrino:
            p = nu2

        child_types = [abs(c.type) for c in tree_preprop.children(p)]
        if  all(ele > 16 for ele in child_types):
            print("Glashow Resonance with two hadrons")
            return(p, tree_preprop.first_child(p))


        for c in tree_preprop.children(p):
            if abs(c.type) not in [11,12,13,14,15,16]:         #13 for muon, 15 for tau
                continue
            return(p,c)
  
