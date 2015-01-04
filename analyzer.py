#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import math,csv,os
from datetime import datetime as dt
from datetime import timedelta as td
RESOURCES_DIR='resources/'

def loadFile(trajactoryfile):
    reader=csv.reader(open(trajactoryfile,"rb"),delimiter=',')
    reader.next();
    x=list(reader)
    return np.array(x).astype('float')
def featurize(filename):
    trajactory = loadFile(filename)
    timeTravel = len(trajactory)
    dirn,filen = ids(filename)

    pathlen = pathlength(trajactory)
    dist = distance(trajactory)
    ar = aspectratio(trajactory)

    sp = speed(trajactory)
    accd,accl = acccomponent(trajactory)
    accd = abs(accd)
    accl = abs(accl)
    return [filen, dirn, timeTravel, pathlen, dist, ar
    , np.median(sp),   np.mean(sp),   np.std(sp),   max(sp)
    , np.median(accd), np.mean(accd), np.std(accd)
    , np.median(accl), np.mean(accl), np.std(accl)]

def featureTitle():
    title = ['filenum', 'drivernum', 'timeTravel', 'pathlen', 'dist', 'ar'
      , 'median(sp)',   'mean(sp)',   'std(sp)',   'max(sp)'
      , 'median(accd)', 'mean(accd)', 'std(accd)'
      , 'median(accl)', 'mean(accl)', 'std(accl)']
    return np.array(title)

def hist(data):
    fig = plt.figure()
    fig.subplots_adjust(hspace=.6,wspace=.3)
    dim = data['data'].shape
    width = int(math.sqrt(dim[1]))
    height = int((dim[1] / width) + (1 if dim[1] % width else 0))
    for d in range(0,dim[1]):
        plt.subplot(width, height, d+1)
        plt.title(data['title'][d])
        plt.hist(data['data'][:,d], 60, normed=0, facecolor='green', alpha=0.75)
    return plt

def plot(data, onefig=False):
    symbs = ['-', '-', '-', '-']
    fig = plt.figure()
    fig.subplots_adjust(hspace=.6,wspace=.3)
    length = len(data)
    width = int(1.61*math.sqrt(length))
    height = int((length / width) + (1 if length % width else 0))
    for d in range(0,length):
        if onefig == False:
            plt.subplot(width, height, d+1)
        points = data[d]['data']
        plt.plot(points[:,0], points[:,1], data[d]['symb'] if 'symb' in data[d] else symbs[d%len(symbs)], label= data[d]['label'] if data[d].has_key('label') else None)
        if data[d].has_key('ylabel'):
            plt.ylabel(data[d]['ylabel'])
        if data[d].has_key('xlabel'):
            plt.xlabel(data[d]['xlabel'])
        if data[d].has_key('title'):
            plt.title(data[d]['title'])
    return plt

# Each element in input is a [x,y] vector. This method normalizes the vector
def normalize(input):
    nms = np.linalg.norm(input, axis=1)
    nms[nms == 0.0] = 1.0
    return (input.T / nms).T
#    return map(lambda vec,nm : [x / nm if nm else x for x in vec], input, nms)

# Each element in each input vectors is a [x,y] point. This method returns a vector of the inner products between each pair of points in the two vector
# The returned vector is truncated to have the length of the shorter input vector
def dot(vec1, vec2):
    return np.array(map(lambda v1,v2 : np.dot(v1,v2), vec1, vec2))

# Each element in each input vectors is a [x,y] point. This method returns a vector of the cross products between each pair of points in the two vector
# The returned vector is truncated to have the length of the shorter input vector
def cross(vec1, vec2):
    return np.array(map(lambda v1,v2 : np.cross(v1,v2), vec1, vec2))

# Decompose the given vector WRT norm vector
# Calculate the perpendicular and parallel component of vector with respect to norm direction.
# norm is assumed a vector of normalized vectors
def decomp(vec1, norm):
    #parallel component
    # For edge case when speed is zero
    # Use map(lambda x,y: x if x else y, norm, vec1) to keep the main component in parallel component
    nv = dot(vec1, norm)
    #perpendicular component, with norm as the x-axis and a right-handed coordinate
    pv = cross(norm, vec1)
    return nv,pv

def acccomponent(trajactory):
    acc = diff(trajactory,2)
    nvec = normalize(diff(trajactory,1))
    return decomp(acc, nvec)

def pathlength(trajactory):
    dd = diff(trajactory,1)
    return sum(np.linalg.norm(dd, axis=1))

def distance(trajactory):
    dd = np.diff([trajactory[0], trajactory[-1]],axis=0)
    return np.linalg.norm(dd)

def diff(data, n=1):
    return np.append(np.diff(data,n=n,axis=0), [[0., 0.]] * n,0)

def speed(trajactory):
    return np.linalg.norm(diff(trajactory,1), axis=1)

def aspectratio(trajactory):
    eigvals, eigvecs = np.linalg.eig(np.cov(trajactory, rowvar=0))
    return max(eigvals) / min(eigvals)

# returns crv,vel,indx (curvature, velocity, and the indexes where speed is non-zero)
def curvature(trajactory):
    sp = speed(trajactory)
    indx = np.where(sp > 0)[0]
    sp = sp[indx]
    vel = np.diff(trajactory,axis=0)[indx]
    acc = np.append(np.diff(trajactory,n=2,axis=0),[[0,0]],0)[indx]
    
    crv = (acc[:,1] * vel[:,0] - acc[:,0] * vel[:,1]).T / (sp ** 3)
    return crv.T,vel,sp,indx

#Find the angle between two vectors, in radians
# Angle is measured in counter-clockwise sense and will be negative if vec2 is on the right of vec1
def angle(vec1, vec2):
#    print 'vec1,vec2'
#    print vec1,vec2
    d = np.cross(vec1,vec2)
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
#    print 'd,n1,n2'
#    print d,n1,n2
    sin = d / n1 / n2
    if abs(sin) > 1.:
        sin = 1. * np.sign(sin)
    return math.asin(sin)

def curvfit(crv):
    abscrv = abs(crv)
    print 'crv segment = {}'.format(crv)
    mcrv = max(abscrv) / 2.
    idxs = np.where(abscrv >= mcrv)[0]
    print 'maxcrv = {}, idxs = {}'.format(mcrv,idxs)
    return mcrv,np.median(idxs)

#Definition of a turn:
#The arc degree > 60 && the arc length < 50 meter
#Both angle and curvature can be negative/positive, so we limit the absolute value of both
# Algorithm 'inchworm algorithm' send an inchworm down the path
# start the segment with zero
# Loop: over the points on the path (grow  the segment one at a time)
#   if the segment contains a turn or a part of it 
#       make the segment the full length MAX_LEN
#       fit the turn out using a turn model
#       mark turn center and the turn angle
#   elif the segment > MAX_LEN
#       reset the segment to be the latter half of itself, i.e., from the midpoint to the end of the original segment
#   else
#       grow segment by one
# 
def turns(trajactory):
    MAX_LEN = 20.
#    MAX_CRV = 50.
#    MIN_CRV = .1
    MIN_TURN = math.pi / 6.
#    MAX_TURN = math.pi * 1.1
    crv,vel,sp,indx = curvature(trajactory)
    cumlen = np.cumsum(sp)
    idxseg = [0,0]
    tidx = []
    while idxseg[1] < len(crv) :
        ang = angle(vel[idxseg[0],:],vel[idxseg[1],:])
        if abs(ang) > MIN_TURN:
            mcrv,centerindx = curvfit(crv[idxseg[0]:idxseg[1]])
            centerindx += idxseg[0]
            tidx.append(centerindx)
            print 'centerindx = {}, curv={}, angle={}'.format(centerindx, mcrv,ang)
            plt.plot(crv[idxseg[0]:idxseg[1]], label='Curvature')
#            print vel[idxseg[0]:idxseg[1], :]
            plt.plot(vel[idxseg[0]:idxseg[1], :], label='velocity')
            plt.plot(sp[idxseg[0]:idxseg[1]], label='speed')
            plt.legend(shadow=True, fancybox=True, loc='upper left')
            plt.show()
            idxseg[0] = idxseg[1]
        elif cumlen[idxseg[0]] - cumlen[idxseg[1]] > 2 * MAX_LEN:
            idxseg[0] = (idxseg[0] + idxseg[1])/2
        else:
            idxseg[1] += MAX_LEN
    print tidx
    return tidx

def signzones(arr):
    sz = np.sign(arr)
    bsz = np.append([1],np.diff(sz))
    return np.where(bsz != 0)[0]

# Return an array of boolean type indicating a change lane event
# 
def changelane(trajactory):
    crv,vel,sp,indx = curvature(trajactory)
    path = trajactory[indx]


def pathiter(basepath, func):
    if os.path.isfile(basepath) and basepath.endswith('.csv'):
        func(basepath)
    elif os.path.isdir(basepath):
        for p in os.listdir(basepath):
            pathiter(os.path.join(basepath,p),func)

def ids(filename):
    filewithoutext = os.path.splitext(filename)[0]
    dirn = int(os.path.basename(os.path.dirname(filewithoutext)))
    filen = int(os.path.basename(filewithoutext))
    return dirn,filen

def testCurvature(basepath):
    trajactory = loadFile(basepath)
    crv,vel,sp,indx = curvature(trajactory)
    print crv
    plt = plot([
        {'data' : np.array(zip(indx, crv)), 'title' : 'Curvature', 'ylabel' : 'Curvature', 'xlabel' : 'index', 'label' : 'Curvature'},
        {'data' : trajactory, 'title' : 'Trajactory', 'ylabel' : 'Y', 'xlabel': 'X', 'label' : 'Path'},
        {'data' : np.array(zip(indx, speed(trajactory))), 'title' : 'Velocity', 'ylabel' : 'Vy', 'xlabel': 'Vx', 'label' : 'Velocity'},
        ])
    outfile = genOutFilename(basepath, 'curvature', 'pdf')
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()

def genOutFilename(basepath, prefix, ext):
    fn,dn = ids(basepath)
    imgname = str(fn)+'-'+str(dn)
    return os.path.join(RESOURCES_DIR, imgname+'-'+prefix+'.'+ext)

def testturns(trajactoryfile):
    trajactory = loadFile(trajactoryfile)
    tindx = turns(trajactory)
    plt = plot([
        {'data' : trajactory, 'title' : 'Trajactory', 'ylabel' : 'Y', 'xlabel': 'X', 'label' : 'Path', 'symb': '.'},
        {'data' : trajactory[tindx], 'title' : 'Turns', 'ylabel' : 'Vy', 'xlabel': 'Vx', 'label' : 'Velocity', 'symb': 'o'},
        ], onefig = True)
    outfile = genOutFilename(trajactoryfile, 'turns', 'pdf')
    print outfile
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()

def trajSpAcc(trajactoryfile):
    trajactory = loadFile(trajactoryfile)

    pathlen = pathlength(trajactory)
    dist = distance(trajactory)
    ar = aspectratio(trajactory)
    print 'Distance = {}, total path length = {}, aspect ratio = {}'.format(dist, pathlen, ar)

    sp = speed(trajactory)
    accl,accd = acccomponent(trajactory)

    filename = os.path.basename(trajactoryfile)
    # acceleration in parallel and perpendicular to the direction of moving
    plt = plot([
        {'data' : trajactory, 'title' : 'Trajactory', 'ylabel' : 'Y', 'xlabel': 'X', 'label' : 'Path'},
        {'data' : np.array(zip(range(0, len(sp)), sp)), 'title' : 'Velocity', 'ylabel' : 'Vy', 'xlabel': 'Vx', 'label' : 'Velocity'},
        {'data' : np.array(zip(range(0, len(accl)), accl)), 'title' : 'Acceleration', 'ylabel' : 'Ay', 'xlabel': 'Ax', 'label' : 'Acceleration'},
        {'data' : np.array(zip(range(0, len(accd)), accd)), 'title' : 'Swaying acc', 'ylabel' : 'Ad', 'xlabel' : 'Al', 'label' : 'Acceleration-sway'},
        ])
    outfile = genOutFilename(trajactoryfile, 'traj_sp_acc', 'pdf')
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()

    crv,vel,sp,indx = curvature(trajactory)
    plt = plot([
        {'data' : np.array(zip(indx, crv)), 'title' : 'Curvature', 'ylabel' : 'Curvature', 'xlabel' : 'index', 'label' : 'Curvature'},
        {'data' : trajactory, 'title' : 'Trajactory', 'ylabel' : 'Y', 'xlabel': 'X', 'label' : 'Path'},
        {'data' : np.array(zip(indx, speed(trajactory))), 'title' : 'Velocity', 'ylabel' : 'Vy', 'xlabel': 'Vx', 'label' : 'Velocity'},
        ])
    outfile = genOutFilename(trajactoryfile, 'curvature', 'pdf')
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()
if __name__ == '__main__':
    if not os.path.exists(RESOURCES_DIR):
        os.makedirs(RESOURCES_DIR)
    import sys
    trajactoryfile = os.path.realpath(sys.argv[1])
    pathiter(trajactoryfile,testCurvature)
