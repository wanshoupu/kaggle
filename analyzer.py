#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import datetime as dt
from datetime import timedelta as td
RESOURCES_DIR='resources/'

def loadFile(trajactoryfile):
    import csv
    reader=csv.reader(open(trajactoryfile,"rb"),delimiter=',')
    reader.next();
    x=list(reader)
    return np.array(x).astype('float')

def hist(data):
    fig = plt.figure()
    fig.subplots_adjust(hspace=.6,wspace=.3)
    dim = data['data'].shape
    import math
    width = int(math.sqrt(dim[1]))
    height = int((dim[1] / width) + (1 if dim[1] % width else 0))
    for d in range(0,dim[1]):
        plt.subplot(width, height, d+1)
        plt.title(data['title'][d])
        n, bins, patches = plt.hist(data['data'][:,d], 60, normed=0, facecolor='green', alpha=0.75)
    return plt

def plot(data):
    symbs = ['-', '-', '-', '-']
    fig = plt.figure()
    fig.subplots_adjust(hspace=.6,wspace=.3)
    length = len(data)
    import math
    width = int(1.61*math.sqrt(length))
    height = int((length / width) + (1 if length % width else 0))
    for d in range(0,length):
        plt.subplot(width, height, d+1)
        points = data[d]['data']
        plt.plot(points[:,0], points[:,1], symbs[d%len(symbs)], label= data[d]['label'] if data[d].has_key('label') else None)
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
    return map(lambda vec,nm : [x / nm if nm else x for x in vec], input, nms)

# Each element in each input vectors is a [x,y] point. This method returns a vector of the inner products between each pair of points in the two vector
# The returned vector is truncated to have the length of the shorter input vector
def dot(vec1, vec2):
    prod = map(lambda v1,v2 : np.dot(v1,v2) if v1 is not None and v2 is not None else None, vec1, vec2)
    return filter(lambda x : x is not None, prod)

# Each element in each input vectors is a [x,y] point. This method returns a vector of the cross products between each pair of points in the two vector
# The returned vector is truncated to have the length of the shorter input vector
def cross(vec1, vec2):
    prod = map(lambda v1,v2 : np.cross(v1,v2) if v1 is not None and v2 is not None else None, vec1, vec2)
    return filter(lambda x : x is not None, prod)

# Decompose the given vector WRT norm vector
# Calculate the perpendicular and parallel component of vector with respect to norm direction.
# norm is assumed a vector of normalized vectors
def decomp(vec1, norm):
    #parallel component
    # For edge case when speed is zero
    # Use map(lambda x,y: x if x else y, norm, vec1) to keep the main component in parallel component
    nv = dot(vec1, map(lambda x: x if x else 1.0, norm))
    #perpendicular component, with norm as the x-axis and a right-handed coordinate
    pv = cross(norm, vec1)
    return np.array([nv, pv]).T

def acccomponent(trajactory):
    acc = np.diff(trajactory,n=2,axis=0)
    nvec = normalize(np.diff(trajactory,axis=0))
    return decomp(acc, nvec)

def pathlength(trajactory):
    dd = np.diff(trajactory,axis=0)
    return sum(np.linalg.norm(dd, axis=1))

def distance(trajactory):
    dd = np.diff([trajactory[0], trajactory[-1]],axis=0)
    return np.linalg.norm(dd)

def speed(trajactory):
    return np.linalg.norm(np.diff(trajactory,axis=0), axis=1)

def aspectratio(trajactory):
    eigvals, eigvecs = np.linalg.eig(np.cov(trajactory, rowvar=0))
    return max(eigvals) / min(eigvals)

if __name__ == '__main__':
    import os
    if not os.path.exists(RESOURCES_DIR):
        os.makedirs(RESOURCES_DIR)
    import sys
    trajactoryfile = sys.argv[1]

    #datetime
    trajactory = loadFile(trajactoryfile)
    pathlen = pathlength(trajactory)
    dist = distance(trajactory)
    ar = aspectratio(trajactory)
    print 'Distance = {}, total path length = {}, aspect ratio = {}'.format(dist, pathlen, ar)

    sp = speed(trajactory)
    acc = acccomponent(trajactory)
    print 'Speed = {}, acceleration = {}'.format(sp, acc)

    filename = os.path.basename(trajactoryfile)
    # acceleration in parallel and perpendicular to the direction of moving
    plt = plot([
        {'data' : trajactory, 'title' : 'Trajactory', 'ylabel' : 'Y', 'xlabel': 'X', 'label' : 'Path'},
        {'data' : np.diff(trajactory,axis=0), 'title' : 'Velocity', 'ylabel' : 'Vy', 'xlabel': 'Vx', 'label' : 'Velocity'},
        {'data' : np.diff(trajactory,n=2,axis=0), 'title' : 'Acceleration', 'ylabel' : 'Ay', 'xlabel': 'Ax', 'label' : 'Acceleration'},
        {'data' : acc, 'title' : 'Projected Acceleration', 'ylabel' : 'Ad', 'xlabel' : 'Al', 'label' : 'Acceleration-projected'},
        ])
    plt.savefig(RESOURCES_DIR+'traj_sp_acc.png', bbox_inches='tight')

