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
    nms = [np.linalg.norm(vec) for vec in input]
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
    nms = [np.linalg.norm(v) for v in vec1]
    #parallel component
    # For edge case when speed is zero
    # Use map(lambda x,y: x if x else y, norm, vec1) to keep the main component in parallel component
    nv = dot(vec1, map(lambda x,y: x if x else y, norm, vec1))
    #perpendicular component, with norm as the x-axis and a right-handed coordinate
    pv = cross(norm, vec1)
    return np.array([nv, pv]).T

def acccomponent(trajactory):
    acc = np.diff(trajactory,n=2,axis=0)
    nvec = normalize(np.diff(trajactory,axis=0))
    return decomp(acc, nvec)

if __name__ == '__main__':
    import os
    if not os.path.exists(RESOURCES_DIR):
        os.makedirs(RESOURCES_DIR)
    import sys
    trajactoryfile = sys.argv[1]

    #datetime
    trajactory = loadFile(trajactoryfile)

    filename = os.path.basename(trajactoryfile)
    # acceleration in parallel and perpendicular to the direction of moving
    acc = acccomponent(trajactory)
    plt = plot([
        {'data' : trajactory, 'title' : 'Trajactory', 'ylabel' : 'Y', 'xlabel': 'X', 'label' : 'Path'},
        {'data' : np.diff(trajactory,axis=0), 'title' : 'Velocity', 'ylabel' : 'Vy', 'xlabel': 'Vx', 'label' : 'Velocity'},
        {'data' : np.diff(trajactory,n=2,axis=0), 'title' : 'Acceleration', 'ylabel' : 'Ay', 'xlabel': 'Ax', 'label' : 'Acceleration'},
        {'data' : acc, 'title' : 'Projected Acceleration', 'ylabel' : 'Ad', 'xlabel' : 'Al', 'label' : 'Acceleration-projected'},
        ])
    plt.savefig(RESOURCES_DIR+'demand_curve.png', bbox_inches='tight')

