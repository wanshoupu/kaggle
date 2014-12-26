from analyzer import loadFile,pathlength,distance,aspectratio,speed,acccomponent,hist,append,ids
import numpy as np
import os,threading

def featurize(filename):
    trajactory = loadFile(filename)
    timeTravel = len(trajactory)
    dirn,filen = ids(filename)

    pathlen = pathlength(trajactory)
    dist = distance(trajactory)
    ar = aspectratio(trajactory)

    sp = speed(trajactory)
    accd,accl = acccomponent(trajactory)
    return [filen,dirn, timeTravel, pathlen, dist, ar
    , np.median(sp),   np.mean(sp),   np.std(sp),   max(sp)
    , np.median(accd), np.mean(accd), np.std(accd)
    , np.median(accl), np.mean(accl), np.std(accl)]

# Processing of all files in a basepath (in a single thread)
#basepath may be a directory, or a file. vec must be a list or tuple
def pathiter(basepath, vec):
    if os.path.isfile(basepath) and basepath.endswith('.csv'):
    #    print 'processing file {}'.format(basepath)
        vec.append(featurize(basepath))
    elif os.path.isdir(basepath):
        for p in os.listdir(basepath):
            pathiter(os.path.join(basepath,p), vec)

#process the group of files under basepath in a batch
def pathgroupiter(basepath, files, vec):
    for f in files:
        print '{} processes file {}...'.format(threading.current_thread().getName(), f)
        pathiter(os.path.join(basepath,f), vec)

def partlistdir(basepath, fanout):
    paths = os.listdir(basepath)
    numpaths = len(paths)
    chunksize = 1000 if numpaths < fanout else (numpaths / fanout)
    for i in xrange(0, len(paths), chunksize):
        yield paths[i:i+chunksize]

# Concurrent processing of all files in a basepath
#basepath must be a directory, vec may be a list
def concurpathiter(basepath, datafile, fanout):
    if os.path.isfile(basepath):
        vec = []
        pathiter(basepath, vec)
        append(datafile, vec)

    threads = {}
    import time
    from random import randint
    for path1 in partlistdir(basepath, fanout):
        vec = []
        t = threading.Thread(target=pathgroupiter, args = (basepath, path1, vec))
        print '{} is assigned files {}:{}'.format(t.getName(), basepath, path1)
        time.sleep(randint(0,5))
        t.start()
        threads[t] = vec

    #wait for all threads to terminate
    while threads:
        remainingThreads = {}
        for t in threads:
            vec = threads[t]
            if t.isAlive():
                remainingThreads[t]= vec
            else:
                append(datafile,vec)
        threads = remainingThreads


RESOURCES_DIR = 'resources'

if __name__ == '__main__':
    if not os.path.exists(RESOURCES_DIR):
        os.makedirs(RESOURCES_DIR)
    import sys
    basepath = os.path.realpath(sys.argv[1])
    fanout = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    vec = []
    if not os.path.exists(basepath):
        print "No such file or path: {}".format(basepath)
    else:
        datafile = os.path.join(RESOURCES_DIR,"data.csv")
        if os.path.exists(datafile):
            os.remove(datafile)
        concurpathiter(basepath, datafile, fanout)
    title = ['filenum', 'drivernum', 'pathlen', 'dist', 'ar'
      , 'median(sp)',   'mean(sp)',   'std(sp)',   'max(sp)'
      , 'median(accd)', 'mean(accd)', 'std(accd)'
      , 'median(accl)', 'mean(accl)', 'std(accl)']

    vec = np.genfromtxt(datafile, delimiter=',')
    plt = hist({'data':vec, 'title':title})
    plt.savefig(os.path.join(RESOURCES_DIR,'hist.png'), bbox_inches='tight')
