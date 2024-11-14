import scipy as sp
import numpy as np
from numpy import linalg as LA
from scipy.linalg import block_diag
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import pyplot as plt
import pandas as pd
import sys
np.set_printoptions(threshold=np.inf)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rc
import statistics as stat

###################################################################
######## functions ##########
###################################################################




def ReadSpinFile(spinfile):
    #data = np.ndarray((len(spinfiles),Nspins,3))
    #SWamplitude = np.ndarray((len(spinfiles),Nspins),dtype=complex)
    
    d = pd.read_csv(spinfile, sep = '\s+', header = None, index_col = False,skiprows=1) #first line is number of spins
        #print(d)
    Nspins = len(d)
    data = np.empty((Nspins, 3))
    data[:,0] = np.copy(d[0])   #Xvalue
    data[:,1] = np.copy(d[1])   #Yvalue
    data[:,2] = np.copy(d[2])   #Zvalue
        
    return data

    
def ReadCoordinateFile(coordFile):
    d = pd.read_csv(coordFile, sep = '\s+', header = None, index_col = False,skiprows=1) #first line is number of spins
        #print(d)
    Nspins = len(d)
    coord = np.empty((Nspins,4))
    coord[:,0] = np.copy(d[0])  #type
    coord[:,1] = np.copy(d[2])  #Xcoord
    coord[:,2] = np.copy(d[3])  #Ycoord
    coord[:,3] = np.copy(d[4])  #Zcoord
    
    # for coord files from spin pumping module: just 3 columns with x,y,z comp, no type provided, so coord[:,0] is dummy here
    #coord[:,0] = np.copy(d[0]) #type
    #coord[:,1] = np.copy(d[1]) #Xcoord
    #coord[:,2] = np.copy(d[2])     #Ycoord
    #coord[:,3] = np.copy(d[3])
    
    return coord,Nspins
    
def plotSpinconfig(coordFile,spinfile):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #fig = plt.figure(figsize=plt.figaspect(0.5)*5) #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)

    coord,Nspins = ReadCoordinateFile(coordFile)
    data = ReadSpinFile(spinfile)
    
    
    magnify = 1.0# 1e2
    #print(np.amax(data[1][0]))
    #stop
    
    #m2 = np.divide(0.4,np.amax(data[1][0]))
    #m3 = np.divide(0.4,np.amax(data[1][1]))
    
    #colors with angle
    #cmap = cm.get_cmap('hsv')
    
    colors = plt.get_cmap('hsv')(np.linspace(0, 1, 360))
    print(colors)
    
    for spin in range(Nspins):
        
        x=data[spin][0]
        y=data[spin][1]
        z=data[spin][2]
        print(x,y)
        alpha = 180*np.angle(x*100+1j*y*100)/np.pi#np.arctan(np.divide(x,y))
        phi = 180*np.angle(y+1j*z)/np.pi
        print('alpha:',alpha)
        
        angle = alpha
        rgb = (1,0,0)# (0.5+0.5*magnify*np.abs(x),0.5+0.5*magnify*np.abs(y),z)
        #rgb2 = ((1.-z)*np.abs(2e5*x),(1.-z)*np.abs(2e5*y),(1.-z)*2e6*np.abs(x*y))
        print('spin,values:',spin,data[spin])
        ax.quiver(coord[spin][1],coord[spin][2],coord[spin][3],magnify*data[spin][0],magnify*data[spin][1],data[spin][2],arrow_length_ratio=0.3,linewidth=4,color=rgb)#colors[int(angle)])#,color=rgb)##color=(0.5+m2*data[spin][0],0.0,0.5-m3*data[spin][1]))#,pivot='middle') ,length=1.0, normalize=True,
        
        
        #ax.scatter(coord[spin][1]+magnify*x,coord[spin][2]+magnify*y, zs=-0.5, zdir='z', color='black')
        ax.scatter(coord[spin][1],coord[spin][2], zs=0, zdir='z', color='green')
        #ax.scatter(coord[spin][1],coord[spin][2], zs=np.sqrt((data[spin][0])**2+(data[spin][1])**2+(data[spin][2])**2), zdir='z', color='black')
        x,y=0,0
        
    #ax.quiver(0,-0.5,0,1,0,0,color='red')
    #ax.quiver(0,-0.5,0,0,1,0,color='blue')
    #ax.quiver(0,-0.5,0,0,0,5,color='green')
    
    #time = np.linspace(0,len(SWamplitude),len(SWamplitude))
    #for f in range(len(SWamplitude)):
    #ax.plot(time,np.real(SWamplitude[:,0]),c='blue')
    #ax.set_ylim(-0.2,1.2)
    #ax.set_xlim(0,4)
    ax.set_xlabel('Sx')
    ax.set_ylabel('Sy')
    ax.set_zlabel('Sz')
    #ax.set_zlim(-1,1)
    
    #plt.show()
    
def plotXY(spinfiles):
    fig = plt.figure()
    for t in range(len(spinfiles)):
        data = ReadSpinFile(spinfiles[t])
        plt.scatter(t,data[1][0], label='x')#,color='blue') #X comp of spin 1
        plt.scatter(t,data[1][1], label='y')#,color='blue') #X comp of spin 1
        plt.scatter(t,data[1][2], label='z')#,color='blue') #X comp of spin 1
        plt.scatter(t,data[359][0], label='y',color='green')
    plt.show()
###################################################################
######## main ##########
###################################################################

def DispArticleSetupPlot(coordfile,spinfile):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    Lx,Ly,Lz = 60,60,1
    ax.quiver(0,0,0,1,0,0,color='gray',linewidth=2,length=Lx*1.1,arrow_length_ratio=0.01)
    ax.quiver(0,0,0,0,1,0,color='gray',linewidth=2,length=Ly*1.1,arrow_length_ratio=0.01)
    ax.quiver(0,0,0,0,0,1,color='gray',linewidth=2,length=Lz*1.1,arrow_length_ratio=0.01)
    ax.quiver(Lx/2,Ly/2,0,0,5,1,color='lime',linewidth=5,length=1,arrow_length_ratio=0.2)
    ax.text(Lx*1.2,0,0,r'$x$',color='gray')
    ax.text(0,Ly*1.2,0,r'$y$',color='gray')
    ax.text(0,0,Lz*1.15,r'$z$',color='gray')
    ax.text(Lx/2,Ly/2,1,r'$DMI$',color='lime')
    
    coord,Nspins = ReadCoordinateFile(coordfile)
    data = ReadSpinFile(spinfile)
    
    
    magnify = 1.0# 1e2
    
    
    colors = plt.get_cmap('hsv')(np.linspace(0, 1, 360))
    
    
    for spin in range(Nspins):
        
        x=data[spin][0]
        y=data[spin][1]
        z=data[spin][2]
        print(x,y)
        alpha = 180*np.angle(x*100+1j*y*100)/np.pi#np.arctan(np.divide(x,y))
        phi = 180*np.angle(y+1j*z)/np.pi
        print('alpha:',alpha)
        
        angle = alpha
        rgb = 'blue'#(0.5+0.5*magnify*np.abs(x),0.5+0.5*magnify*np.abs(y),z)
        #rgb2 = ((1.-z)*np.abs(2e5*x),(1.-z)*np.abs(2e5*y),(1.-z)*2e6*np.abs(x*y))
        print('spin,values:',spin,data[spin])
        #ax.quiver(coord[spin][1],coord[spin][2],coord[spin][3],magnify*data[spin][0],magnify*data[spin][1],-1*data[spin][2],arrow_length_ratio=0.3,linewidth=2,color=rgb,length=0.5,normalize=True)
        ax.quiver(coord[spin][1],coord[spin][2],coord[spin][3],np.zeros(Nspins),np.zeros(Nspins),np.ones(Nspins),length=0.5,normalize=True,linewidth=2)
        
        
        ax.scatter(coord[spin][1],coord[spin][2], zs=0, zdir='z', color='green')
        
        x,y=0,0
    
    ax.set_xlabel('Sx')
    ax.set_ylabel('Sy')
    ax.set_zlabel('Sz')
    ax.set_zlim(-1,1)
    
    plt.show()
    
    #look all ugly, rather use tikz
    
def plotLatticeOnly(coordfile):
    coord,Nspins = ReadCoordinateFile(coordfile)
    
    plt.scatter(coord[:,1],coord[:,2],c=coord[:,3])
    plt.show()

def plot3D(coordfile,spinfile):
    coord,Nspins = ReadCoordinateFile(coordfile)    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    data = ReadSpinFile(spinfile)
    ax.scatter(coord[:,1],coord[:,2],coord[:,3])
    
    for s in range(Nspins):
        #projection
        '''if (coord[s,3]<10):
            ax.scatter(coord[s,1],coord[s,2],np.zeros(coord[s,3].shape),color='blue',alpha=0.3)
            ax.scatter(coord[s,1],coord[s,2],coord[s,3],color='blue')
        else:
            ax.scatter(coord[s,1],coord[s,2],np.zeros(coord[s,3].shape),color='black',marker='x')
            ax.scatter(coord[s,1],coord[s,2],coord[s,3],color='grey')'''
        ax.quiver(coord[s,1],coord[s,2],coord[s,3],data[s,0],data[s,1],data[s,2],color='red',length=2,linewidth=2)
    #plt.show()

def main():
    coordfile = sys.argv[1]
    datafiles = sys.argv[2:]
    #plotLattice3D(coordfile)
    #plotLatticeOnly(coordfile)
    #stop
    #print(coordfile, datafiles)
    
    
    for i in range(len(datafiles)):
        ifile = datafiles[i]
        plot3D(coordfile,ifile)
        #plotSpinconfig(coordfile,ifile)
        #DispArticleSetupPlot(coordfile, ifile)
        plt.show()
        #plt.savefig('Plots/'+str(i).zfill(4)+'.png')
    
    #plotXY(datafiles)
if __name__ == '__main__':
    main()
