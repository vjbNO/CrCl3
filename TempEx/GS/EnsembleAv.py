import numpy as np
import glob
#from concurrent.functions import ProcessPoolExecutor



def timeav(runs):

        for r in runs:
                #print('start run', r)
                #foldername = './run_'+str(r)+'/ProcessedData/'
                foldername = './ProcessedData/'
                print(foldername)
                datafilesUp = np.sort(glob.glob(foldername+'Hexdata_UpperLayer*.txt')) 
                datafilesDown = np.sort(glob.glob(foldername+'Hexdata_LowerLayer*.txt'))
                print(datafilesUp,datafilesDown)
                
                d0 = np.loadtxt(datafilesUp[0])
                Nx = len(d0)
                Ny = len(d0[0])
                tlen = len(datafilesUp)
                dup = np.zeros((tlen,Nx,Ny))
                ddown = np.zeros((tlen,Nx,Ny))

                for t in range(10,len(datafilesUp)): #skip first few for steady state
                        dup[t,:,:] = np.loadtxt(datafilesUp[t])
                        ddown[t,:,:] = np.loadtxt(datafilesDown[t])

                timeavUp = np.mean(dup,axis=0)
                timeavDown = np.mean(ddown,axis=0)
                #std = np.std(d,axis=0)
                np.savetxt(foldername+'TimeAvUp_run_'+str(r)+'.txt',timeavUp)
                np.savetxt(foldername+'TimeAvDown_run'+str(r)+'.txt',timeavDown)

def ensembleav(runs):
        foldername = './EnsembleAv/'
        datafiles = np.sort(glob.glob(foldername+'TimeAv_run*')) #sys.argv[1:]
        print(datafiles)
        d0 = np.loadtxt(datafiles[0])
        Nx = len(d0)
        Ny = len(d0[0])
        d = np.zeros((len(runs),Nx,Ny))
        for r in range(len(runs)):
                d[r,:,:] = np.loadtxt(datafiles[r])

        ensembleav = np.mean(d,axis=0)

        np.savetxt('EnsembleAv/EnsembleAvAfterTav.txt',ensembleav)

if __name__ == '__main__':
        runs = np.arange(1,2,1)
        print(runs)
        timeav(runs)
        #ensembleav(runs)
