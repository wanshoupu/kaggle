from analyzer import loadFile,pathlength,distance,aspectratio,speed,acccomponent,hist
import numpy as np
import os

def featurize(filename):
    #datetime
    trajactory = loadFile(filename)
    pathlen = pathlength(trajactory)
    dist = distance(trajactory)
    ar = aspectratio(trajactory)

    sp = speed(trajactory)
    acc = acccomponent(trajactory)
    accd = abs(acc.T[0])
    accl = abs(acc.T[1])
    return [pathlen, dist, ar
    , np.median(sp),   np.mean(sp),   np.std(sp),   max(sp)
    , np.median(accd), np.mean(accd), np.std(accd)
    , np.median(accl), np.mean(accl), np.std(accl)]

def pathiter(basepath, vec):
  if os.path.isfile(basepath):
    vec.append(featurize(basepath))
  elif os.path.isdir(basepath):
    for p in os.listdir(basepath):
      pathiter(os.path.join(basepath,p), vec)

RESOURCES_DIR = 'resources'

if __name__ == '__main__':
  if not os.path.exists(RESOURCES_DIR):
      os.makedirs(RESOURCES_DIR)
  import sys
  basepath = sys.argv[1]
  vec = []
  if not os.path.exists(basepath):
    print "No such file or path: {}".format(basepath)
  else:
    pathiter(basepath, vec)
  vec = np.asmatrix(vec)
  np.savetxt(os.path.join(RESOURCES_DIR,"foo.csv"), vec, delimiter=",")
  title = ['pathlen', 'dist', 'ar'
    , 'median(sp)',   'mean(sp)',   'std(sp)',   'max(sp)'
    , 'median(accd)', 'mean(accd)', 'std(accd)'
    , 'median(accl)', 'mean(accl)', 'std(accl)']
  plt = hist({'data':vec, 'title':title})
  plt.savefig(os.path.join(RESOURCES_DIR,'hist.png'), bbox_inches='tight')
