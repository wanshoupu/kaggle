from analyzer import hist,ids,pathiter,featurize,featureTitle
import numpy as np
import os,threading,csv
#import math,csv,os

#process the group of files under basepath in a batch
def pathgroupiter(basepath, files, vec):
    def proc(basepath):
        vec.append(featurize(basepath))

    for f in files:
        print '{} processes file {}...'.format(threading.current_thread().getName(), f)
        pathiter(os.path.join(basepath,f), proc)

def partlistdir(basepath, fanout):
    paths = os.listdir(basepath)
    numpaths = len(paths)
    chunksize = 1000 if numpaths < fanout else (numpaths / fanout)
    for i in xrange(0, len(paths), chunksize):
        yield paths[i:i+chunksize]

#append data to file
def append(file, data):
    if data:
        with open(file, 'a') as fp:
            a = csv.writer(fp, delimiter=',');
            a.writerows(data);

# Concurrent processing of all files in a basepath
#basepath must be a directory, vec may be a list
def concurpathiter(basepath, datafile, fanout):
    if os.path.isfile(basepath):
        vec = []
        def proc(basepath):
            vec.append(featurize(basepath))
        pathiter(basepath, proc)
        append(datafile, vec)
        return

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
    for t in threads:
        t.join()
        vec = threads[t]
        append(datafile,vec)

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
        title = featureTitle()
        vec = np.genfromtxt(datafile, delimiter=',')
        drivernums = set(vec[:,1])
        for d in drivernums:
            plt = hist({'data':vec[vec[:,1] == d], 'title':title+'-'+str(d)})
            plt.savefig(os.path.join(RESOURCES_DIR,'hist-'+str(d)+'.png'), bbox_inches='tight')
