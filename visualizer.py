from analyzer import hist,featureTitle
import numpy as np
import sys,os

def visualize(datafile):
    datafile = os.path.realpath(datafile)
    title = featureTitle()
    vec = np.genfromtxt(datafile, delimiter=',')
    drivernums = set(vec[:,1])
    for d in drivernums:
        data = vec[vec[:,1] == d][:,2:]#[:,np.array([10,11,13,14])]
        plt = hist({'data':data, 'title':title[2:]#[np.array([10,11,13,14])]
            })
        figname = os.path.join(os.path.dirname(datafile),'hist-'+str(d)+'.pdf')
        print 'saving {} ...'.format(figname)
        plt.savefig(figname, bbox_inches='tight')

if __name__ == '__main__':
    visualize(sys.argv[1])