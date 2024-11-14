import scipy as sp
import numpy as np
import time
import pandas as pd
import glob
import heapq
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor



def Definecenters(pos):
    #dx,dy from lower layer
    shortdy = np.abs(pos[3,1]-pos[2,1])/2.0 #from center to left up
    longdy = np.abs(pos[1,1]-pos[4,1])/2.0 #from center to upper end
    dx = np.abs(pos[1,0]-pos[2,0])
    
    firstcenterDown= [pos[4,0],pos[4,1]] #first center down
    firstcenterOffsetDown = [pos[7,0],pos[7,1]]
    firstcenterUp = [pos[1,0],pos[1,1]] #first center
    firstcenterOffsetUp = [pos[2,0],pos[2,1]]
    
    HexDistX = 2.0*dx*1.001 #correction factor
    HexDistY = 2.0*np.abs(firstcenterDown[1]-firstcenterOffsetDown[1])
    
    lastcenterDown = [-1*firstcenterDown[0],-1*firstcenterDown[1]]
    lastcenterOffsetDown = [-1*firstcenterOffsetDown[0],-1*firstcenterOffsetDown[1]]
    lastcenterUp = [-1*firstcenterUp[0],-1*firstcenterUp[1]]
    lastcenterOffsetUp = [-1*firstcenterOffsetUp[0],-1*firstcenterOffsetUp[1]]
    
    # Number of hexagons
    Nx = int((lastcenterUp[0]-firstcenterUp[0])/HexDistX)
    Ny = int((lastcenterUp[1]-firstcenterUp[1])/HexDistY)

    HexPositionsUp = np.mgrid[firstcenterUp[0]:lastcenterUp[0]:HexDistX,firstcenterUp[1]:lastcenterUp[1]:HexDistY]
    HexPosOffsetUp = np.mgrid[firstcenterOffsetUp[0]:lastcenterOffsetUp[0]:HexDistX,firstcenterOffsetUp[1]:lastcenterOffsetUp[1]:HexDistY]
    HexPositionsUp = np.concatenate((HexPositionsUp,np.ones(HexPositionsUp.shape)*2.9375))
    HexPosOffsetUp = np.concatenate((HexPosOffsetUp,np.ones(HexPosOffsetUp.shape)*2.9375))

    HexPositionsDown = np.mgrid[firstcenterDown[0]:lastcenterDown[0]:HexDistX,firstcenterDown[1]:lastcenterDown[1]:HexDistY]
    HexPosOffsetDown = np.mgrid[firstcenterOffsetDown[0]:lastcenterOffsetDown[0]:HexDistX,firstcenterOffsetDown[1]:lastcenterOffsetDown[1]+HexDistY:HexDistY] 
    HexPositionsDown = np.concatenate((HexPositionsDown,np.ones(HexPositionsDown.shape)*-2.9375))
    HexPosOffsetDown = np.concatenate((HexPosOffsetDown,np.ones(HexPosOffsetDown.shape)*-2.9375))


    return Nx,Ny,HexPositionsUp,HexPosOffsetUp,HexPositionsDown,HexPosOffsetDown,dx,shortdy,longdy

def FindData(Nx,Ny,pos_swap,HexPositionsUp,HexPosOffsetUp,HexPositionsDown,HexPosOffsetDown,dx,shortdy,longdy,coorddata):
    #plt.scatter(coorddata[:,0],coorddata[:,1],label='coorddata')
    #plt.scatter(HexPositionsUp[0,:,:],HexPositionsUp[1,:,:],label='hexpos')
    #plt.legend()
    #plt.show()
    #stop
    HexDataUp = np.zeros((Nx,2*Ny,2))
    HexDataDown = np.zeros((Nx,2*Ny,2))
    e=[]
    for nx in range(Nx):
        for ny in range(int(2*Ny)):
            if (ny%2==1):
                centerUp = HexPositionsUp[:,nx,int(ny/2)]
                centerDown = HexPositionsDown[:,nx,int(ny/2)]
            else:
                centerUp = HexPosOffsetUp[:,nx,int(ny/2)]
                centerDown = HexPosOffsetDown[:,nx,int(ny/2)]
            muUp = []
            muDown = []
            
            def searchUp(x2u,y2u,z2u,muUp,countUp):
                positionsUpRounded = ((round(x2u-dx,0),round(y2u+shortdy,0),round(z2u,0)) , (round(x2u-dx,0),round(y2u-shortdy,0),round(z2u,0)) , (round(x2u+dx,0),round(y2u+shortdy,0),round(z2u,0)) , (round(x2u+dx,0),round(y2u-shortdy,0),round(z2u,0)) , (round(x2u,0),round(y2u+longdy+shortdy,0),round(z2u,0)) , (round(x2u,0),round(y2u-longdy-shortdy,0),round(z2u,0)))
                
                for pos in positionsUpRounded:
                    if pos in pos_swap:
                        mul = (pos_swap[pos])
                        muUp.append(mul)
                        countUp +=1
                #print('countUp ',countUp)
                return muUp,countUp
                
            def searchDown(x2d,y2d,z2d,muDown,countDown):
                positionsDownRounded = ((round(x2d-dx,0),round(y2d+shortdy,0),round(z2d,0)) , (round(x2d-dx,0),round(y2d-shortdy,0),round(z2d,0)) , (round(x2d+dx,0),round(y2d+shortdy,0),round(z2d,0)) , (round(x2d+dx,0),round(y2d-shortdy,0),round(z2d,0)) , (round(x2d,0),round(y2d+longdy+shortdy,0),round(z2d,0)) , (round(x2d,0),round(y2d-longdy-shortdy,0),round(z2d,0)))
                
                for posd in positionsDownRounded:
                    if posd in pos_swap:
                        mul = (pos_swap[posd])
                        muDown.append(mul)
                        countDown +=1
                #print('countDown ',countDown)
                return muDown,countDown
            countUp = 0
            countDown = 0
           
            x2u,y2u,z2u = float(centerUp[0]),float(centerUp[1]),float(centerUp[2])
            x2d,y2d,z2d = float(centerDown[0]),float(centerDown[1]),float(centerDown[2])
            muUp,countUp = searchUp(x2u,y2u,z2u,muUp,countUp)
            muDown,countDown = searchDown(x2d,y2d,z2d,muDown,countDown)
            if (countUp<6):
                muUp,countUp = searchUp(x2u-1,y2u,z2u,muUp,countUp)
            if (countUp<6):                
                muUp,countUp = searchUp(x2u+1,y2u,z2u,muUp,countUp)
            if (countUp<6):
                muUp,countUp = searchUp(x2u,y2u+1,z2u,muUp,countUp)
            if (countUp<6):                
                muUp,countUp = searchUp(x2u,y2u-1,z2u,muUp,countUp)
            if (countUp<6):
                muUp,countUp = searchUp(x2u-1,y2u-1,z2u,muUp,countUp)
            if (countUp<6):                
                muUp,countUp = searchUp(x2u+1,y2u+1,z2u,muUp,countUp)
            if (countUp<6):
                muUp,countUp = searchUp(x2u-1,y2u+1,z2u,muUp,countUp)
            if (countUp<6):                
                muUp,countUp = searchUp(x2u+1,y2u-1,z2u,muUp,countUp)
            if (countDown<6):
                muDown,countDown = searchDown(x2d+1,y2d,z2d,muDown,countDown)
            if (countDown<6):                
                muDown,countDown = searchDown(x2d-1,y2d,z2d,muDown,countDown)
            if (countDown<6):
                muDown,countDown = searchDown(x2d,y2d+1,z2d,muDown,countDown)
            if (countDown<6):                
                muDown,countDown = searchDown(x2d,y2d-1,z2d,muDown,countDown)
            if (countDown<6):
                muDown,countDown = searchDown(x2d-1,y2d+1,z2d,muDown,countDown)
            if (countDown<6):                
                muDown,countDown = searchDown(x2d+1,y2d-1,z2d,muDown,countDown)
            if (countDown<6):
                muDown,countDown = searchDown(x2d-1,y2d-1,z2d,muDown,countDown)
            if (countDown<6):                
                muDown,countDown = searchDown(x2d+1,y2d+1,z2d,muDown,countDown)
            if (countUp<6):
                muUp,countUp = searchUp(x2u-2,y2u,z2u,muUp,countUp)
            if (countUp<6):                
                muUp,countUp = searchUp(x2u+2,y2u,z2u,muUp,countUp)
            '''if (countUp<6):
                muUp,countUp = searchUp(x2u,y2u+2,z2u,muUp,countUp)
            if (countUp<6):                
                muUp,countUp = searchUp(x2u,y2u-2,z2u,muUp,countUp)
            if (countUp<6):
                muUp,countUp = searchUp(x2u-2,y2u-2,z2u,muUp,countUp)
            if (countUp<6):                
                muUp,countUp = searchUp(x2u+2,y2u+2,z2u,muUp,countUp)
            if (countUp<6):
                muUp,countUp = searchUp(x2u-2,y2u+2,z2u,muUp,countUp)
            if (countUp<6):                
                muUp,countUp = searchUp(x2u+2,y2u-2,z2u,muUp,countUp)
            '''
            if (countDown<6):
                muDown,countDown = searchDown(x2d+2,y2d,z2d,muDown,countDown)
            if (countDown<6):                
                muDown,countDown = searchDown(x2d-2,y2d,z2d,muDown,countDown)
            '''if (countDown<6):
                muDown,countDown = searchDown(x2d,y2d+2,z2d,muDown,countDown)
            if (countDown<6):                
                muDown,countDown = searchDown(x2d,y2d-2,z2d,muDown,countDown)
            if (countDown<6):
                muDown,countDown = searchDown(x2d-2,y2d+2,z2d,muDown,countDown)
            if (countDown<6):                
                muDown,countDown = searchDown(x2d+2,y2d-2,z2d,muDown,countDown)
            if (countDown<6):
                muDown,countDown = searchDown(x2d-2,y2d-2,z2d,muDown,countDown)
            if (countDown<6):                
                muDown,countDown = searchDown(x2d+2,y2d+2,z2d,muDown,countDown)
            
            if (countUp<6):
                muUp,countUp = searchUp(x2u-1,y2u+2,z2u,muUp,countUp)
            if (countUp<6):                
                muUp,countUp = searchUp(x2u-1,y2u-2,z2u,muUp,countUp)
            if (countUp<6):
                muUp,countUp = searchUp(x2u+1,y2u+2,z2u,muUp,countUp)
            if (countUp<6):                
                muUp,countUp = searchUp(x2u+1,y2u-2,z2u,muUp,countUp)
            '''
            if (countUp<6):
                muUp,countUp = searchUp(x2u-2,y2u+1,z2u,muUp,countUp)
            if (countUp<6):                
                muUp,countUp = searchUp(x2u-2,y2u-1,z2u,muUp,countUp)
            if (countUp<6):
                muUp,countUp = searchUp(x2u+2,y2u-1,z2u,muUp,countUp)
            if (countUp<6):             
                muUp,countUp = searchUp(x2u+2,y2u+1,z2u,muUp,countUp)
            '''
            if (countDown<6):
                muDown,countDown = searchDown(x2d+1,y2d-2,z2d,muDown,countDown)
            if (countDown<6):                
                muDown,countDown = searchDown(x2d+1,y2d+2,z2d,muDown,countDown)
            if (countDown<6):
                muDown,countDown = searchDown(x2d-1,y2d-2,z2d,muDown,countDown)
            if (countDown<6):                
                muDown,countDown = searchDown(x2d-1,y2d+2,z2d,muDown,countDown)
            '''
            if (countDown<6):
                muDown,countDown = searchDown(x2d-2,y2d+1,z2d,muDown,countDown)
            if (countDown<6):                
                muDown,countDown = searchDown(x2d-2,y2d-1,z2d,muDown,countDown)
            if (countDown<6):
                muDown,countDown = searchDown(x2d+2,y2d-1,z2d,muDown,countDown)
            if (countDown<6):                
                muDown,countDown = searchDown(x2d+2,y2d+1,z2d,muDown,countDown)
            
            positions = []
            muUp = muUp
            muDown = muDown
            HexDataUp[nx,ny,0]=np.mean(muUp)
            HexDataUp[nx,ny,1]=countUp
            HexDataDown[nx,ny,0]=np.mean(muDown)
            HexDataDown[nx,ny,1]=countDown
            
    return HexDataUp,HexDataDown


def task(dataname): #what comes in is a singular data file name (string)
    print('now working on dataset ',dataname)
    coorddata = np.loadtxt('atoms-spin-pumping-coords.data',skiprows=1)#sys.argv[2],skiprows=1)
    Nspins = int(np.loadtxt('atoms-spin-pumping-coords.data',max_rows=1))
    print('So many spins:',Nspins)
    
    savedir = './ProcessedData/' 
    
    
    realtime = (dataname.split('-')[3]).split('.')[0] #extract the time stamp from the file name
    print('realtime:',realtime)
    rawdata = np.loadtxt(dataname,skiprows=1)
    
    data = {}

    for i in range(Nspins): #mux is 'key', coords are 'value'. Make mux unique with small marker, because if keys are the same, the item gets replaced!
        data[rawdata[i,2]*(1+1e-15*i)] = (np.round(coorddata[i,0],0),np.round(coorddata[i,1],0),np.round(coorddata[i,2],0))

    orderedY = dict(sorted(data.items(), key=lambda item: item[1][1]))
    orderedYX = dict(sorted(data.items(), key=lambda item: item[1][0]))

    # swap ID and position
    def get_swap_dict(d):
        return {v: k for k, v in d.items()}

    pos_swap = get_swap_dict(orderedYX) #now coords are the keys and mux are values

    Nx,Ny,HexPositionsUp,HexPosOffsetUp,HexPositionsDown,HexPosOffsetDown,dx,shortdy,longdy = Definecenters(coorddata)
    
    print('so many in totoal Nx,Ny:',Nx,Ny)
    HexDataUp,HexDataDown = FindData(Nx,Ny,pos_swap,HexPositionsUp,HexPosOffsetUp,HexPositionsDown,HexPosOffsetDown,dx,shortdy,longdy,coorddata)
    
    #plt.imshow(HexDataDown[:,:,1])
    #plt.imshow(HexDataUp[:,:,1])
    #plt.show()
    #stop
    np.savetxt(savedir+'Hexdata_UpperLayer_time_'+realtime+'.txt',HexDataUp[:,:,0])
    np.savetxt(savedir+'Hexdata_LowerLayer_time_'+realtime+'.txt',HexDataDown[:,:,0])
    

def main():
    datalist = np.sort(glob.glob('atoms-spin-pumping-0000*.data'))
    #task(datalist[0])
    #stop
    with ProcessPoolExecutor(25) as executor:
       executor.map(task,datalist)

if __name__ == '__main__':
    main()
