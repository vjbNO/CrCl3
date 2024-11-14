import numpy as np
import glob
#from concurrent.functions import ProcessPoolExecutor



def timeav(runs):

        for r in runs:
                print('start run', r)
                foldername = './run_'+str(r)+'/ProcessedData/'
                #foldername = './ProcessedData/'
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
                np.savetxt('./EnsembleAv/TimeAvUp_run_'+str(r)+'.txt',timeavUp)
                np.savetxt('./EnsembleAv/TimeAvDown_run'+str(r)+'.txt',timeavDown)

def ensembleav(runs):
        foldername = './EnsembleAv/'
        datafilesUp = np.sort(glob.glob(foldername+'TimeAvUp_run*')) #sys.argv[1:]
        datafilesDown = np.sort(glob.glob(foldername+'TimeAvDown_run*'))
        print(datafilesUp,datafilesDown)
        d0 = np.loadtxt(datafilesUp[0])
        Nx = len(d0)
        Ny = len(d0[0])
        dup = np.zeros((len(runs),Nx,Ny))
        ddown = np.zeros((len(runs),Nx,Ny))
        for r in range(len(runs)):
                dup[r,:,:] = np.loadtxt(datafilesUp[r])
                ddown[r,:,:] = np.loadtxt(datafilesDown[r])

        ensembleavUp = np.mean(dup,axis=0)
        ensembleavDown = np.mean(ddown,axis=0)

        np.savetxt('EnsembleAv/EnsembleAvAfterTav_UpperLayer.txt',ensembleavUp)
        np.savetxt('EnsembleAv/EnsembleAvAfterTav_LowerLayer.txt',ensembleavDown)

if __name__ == '__main__':
        runs = np.arange(1,11,1)
        print(runs)
        timeav(runs)
        ensembleav(runs)
