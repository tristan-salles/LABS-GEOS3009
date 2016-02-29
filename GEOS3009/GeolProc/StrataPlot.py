import os
from vtk import *
import numpy as np
import pandas as pd
import numpy.ma as ma

import matplotlib
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from scipy.interpolate import griddata as scipygrid
from vtk.util.numpy_support import vtk_to_numpy

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

def read_seaLevel(seafile):

    df=pd.read_csv(seafile, sep='\t',header=None)
    SLtime,sealevel = df[0],df[1]

    fig = plt.figure(figsize = (5,10))
    plt.rc("font", size=14)

    ax1 = fig.add_subplot(1, 1, 1)
    minZ = SLtime.min()
    maxZ = SLtime.max()
    minX = sealevel.min()
    maxX = sealevel.max()

    plt.plot(sealevel,SLtime,'o-',color='#6666FF',linewidth=2,label='model 2 sea level fall')

    axes = plt.gca()
    plt.xlim( minX-10, maxX+10 )
    plt.ylim( minZ, maxZ )
    plt.grid(True)
    plt.xlabel('Sea level (m)',fontsize=14)
    plt.ylabel('Time (years)',fontsize=14)
    plt.legend(loc=2, fontsize=12)

    return SLtime,sealevel

def read_VTK(basefile,stratafile,initTime):
    '''
    This function reads strataform basement and stratigraphic VTK files and
    return numpy arrays for:
        - x, y, z coordinates
        - ages
        - mean grain size
    '''

    if os.path.isfile(basefile) and os.access(basefile, os.R_OK):
        print "Basement file exists and is readable"
    else:
        print "Either basement file is missing or is not readable"
        return


    if os.path.isfile(stratafile) and os.access(stratafile, os.R_OK):
        print "Stratigraphic file exists and is readable"
    else:
        print "Either stratigraphic file is missing or is not readable"
        return

    # Load basement surface vtk file as input
    Breader = vtkXMLUnstructuredGridReader()
    Breader.SetFileName(basefile)
    Breader.Update()

    # Get the coordinates of basement surface nodes
    bnodes_vtk_array= Breader.GetOutput().GetPoints().GetData()

    # Get the coordinates of the nodes and their mean grain sizes and ages
    bnodes_nummpy_array = vtk_to_numpy(bnodes_vtk_array)
    bx,by,bz= bnodes_nummpy_array[:,0],bnodes_nummpy_array[:,1],bnodes_nummpy_array[:,2]
    minBase = bz.min()

    dx = bx[1]-bx[0]
    nx = int((bx.max()-bx.min())/dx)+1
    ny = int((by.max()-by.min())/dx)+1

    baseX,baseY,baseZ = bx[nx*ny:],by[nx*ny:],bz[nx*ny:]
    bmgz = np.zeros(len(baseX))
    bage = np.zeros(len(baseX))
    bage.fill(initTime)

    # Load stratigraphic layer vtk file as input
    Sreader = vtk.vtkXMLUnstructuredGridReader()
    Sreader.SetFileName(stratafile)
    Sreader.Update()

    # Get the coordinates of nodes in the mesh
    snodes_vtk_array= Sreader.GetOutput().GetPoints().GetData()
    mgz_vtk = Sreader.GetOutput().GetPointData().GetArray("mean grain size")
    age_vtk = Sreader.GetOutput().GetPointData().GetArray("age smooth")

    # Get the coordinates of the nodes and their mean grain sizes and ages
    snodes_nummpy_array = vtk_to_numpy(snodes_vtk_array)
    sx,sy,sz = snodes_nummpy_array[:,0],snodes_nummpy_array[:,1],snodes_nummpy_array[:,2]

    mgz_numpy_array = vtk_to_numpy(mgz_vtk)
    smgz = mgz_numpy_array*1000. # change from m to mm

    age_numpy_array = vtk_to_numpy(age_vtk)
    sage = age_numpy_array

    # Concatenate basement and stratigraphy arrays
    x = np.hstack((baseX, sx))
    y = np.hstack((baseY, sy))
    z = np.hstack((baseZ, sz))
    mgz = np.hstack((bmgz, smgz))
    age = np.hstack((bage, sage))

    print "Numpy arrays with relevant informations have been created."

    return minBase, x, y, z, mgz, age

def mapData_Reg(nlays,x,y,z,mgz,age,res,simStart,dt):

    nlays += 2

    dx = x[1]-x[0]
    nx = int((x.max()-x.min())/dx)+1
    ny = int((y.max()-y.min())/dx)+1

    # Create the numpy arrays
    xp = np.zeros((nlays*nx*ny))
    yp = np.zeros((nlays*nx*ny))
    zp = np.zeros((nlays*nx*ny))
    mzp = np.zeros((nlays*nx*ny))
    agex = np.zeros((nlays*nx*ny))
    zpos = np.zeros((nlays,nx,ny))
    zpos.fill(-9999.)
    mzpos = np.zeros((nlays,nx,ny))
    mzpos.fill(-9999.)

    # Define the data grid
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)

    # Fill stratigraphic hiatus with underlying stratal values
    for j in range(ny):
        for i in range(nx):
            ids = np.where( (x==xi[i]) & (y==yi[j]))[0]
            ages = age[ids]
            zs = z[ids]
            ms = mgz[ids]
            for l in range(len(ages)):
                layNb = int((ages[l] - simStart)/dt)
                zpos[layNb,i,j] = zs[l]
                mzpos[layNb,i,j] = ms[l]

            for l in range(0,nlays):
                if zpos[l,i,j] < -9998.:
                    zpos[l,i,j] = zpos[l-1,i,j]
                if mzpos[l,i,j] < -9998.:
                    mzpos[l,i,j] = mzpos[l-1,i,j]

    # Transform to 1D arrays
    k = 0
    for l in range(nlays):
        for j in range(ny):
            for i in range(nx):
                xp[k] = x.min()+i*dx
                yp[k] = y.min()+j*dx
                zp[k] = zpos[l,i,j]
                mzp[k] = mzpos[l,i,j]
                k += 1

    # Define the interpolation grid
    # Number of points along X-axis
    nxi = int((x.max() - x.min())/res+1)
    # Number of points along Y-axis
    nyi = int((y.max() - y.min())/res+1)

    # Define linearly spaced data along each axis
    xi = np.linspace(x.min(), x.max(), nxi)
    yi = np.linspace(y.min(), y.max(), nyi)

    # Create the regular mesh
    xi, yi = np.meshgrid(xi, yi, indexing='xy')

    # For each layer interpolate the values on the refined grid
    zi = np.zeros((xi.shape[0],xi.shape[1],nlays))
    mzi = np.zeros((xi.shape[0],xi.shape[1],nlays))
    for l in range(nlays):
        start = l*(nx*ny)
        end = (l+1)*(nx*ny)
        zlayi = ml.griddata(xp[start:end],yp[start:end],zp[start:end],xi,yi,interp='linear')
        zi[:,:,l] = zlayi
        mzlayi = ml.griddata(xp[start:end],yp[start:end],mzp[start:end],xi,yi,interp='linear')
        mzi[:,:,l] = mzlayi

    print "Dataset has been mapped on a regular grid."

    return nlays,xi,yi,zi,mzi,nxi,nyi

def crossXsection(posY,res,xi,yi,nxi,nyi,minZ,slvl):

    # Find the ID of the node along the Y-axis that has the chosen coordinate
    yID = int((posY - yi.min())/res)

    # Define the cross-section line along oX
    Xsec = np.linspace(xi.min(), xi.max(), nxi)

    # The base of the model is fixed to -600 m
    base = np.zeros((Xsec.shape[0]))
    base.fill(minZ-10)

    # Define sea-level elevation
    sl = np.zeros((Xsec.shape[0]))
    sl.fill(slvl)

    return Xsec,base,sl,yID

def crossYsection(posX,res,xi,yi,nxi,nyi,minZ,slvl):

    # Find the ID of the node along the X-axis that has the chosen coordinate
    xID = int((posX - xi.min())/res)

    # Define the cross-section line along oY
    Ysec = np.linspace(yi.min(), yi.max(), nyi)

    # The base of the model is fixed to -600 m
    base = np.zeros((Ysec.shape[0]))
    base.fill(minZ-10)

    # Define sea-level elevation
    sl = np.zeros((Ysec.shape[0]))
    sl.fill(slvl)

    return Ysec,base,sl,xID

def getXmgz(Xsec,zi,xi,mzi,nlays,yID,mgzExtent):

    xx = np.zeros(mzi.shape[0]*mzi.shape[2])
    zz = np.zeros(mzi.shape[0]*mzi.shape[2])
    mgzz = np.zeros(mzi.shape[0]*mzi.shape[2])
    k = 0
    for l in range(0,nlays):
        for i in range(mzi.shape[0]):
            xx[k] = xi[yID,i]
            zz[k] = zi[yID,i,l]
            mgzz[k] = mzi[yID,i,l]
            k += 1

    # define grid.
    zi2 = np.linspace(mgzExtent[0],mgzExtent[1],mgzExtent[2])

    # grid the data.
    mgzi2 = griddata(xx, zz, mgzz, Xsec, zi2, interp='linear')
    # Mask points above top surface
    for j in range(len(Xsec)):
        for i in range(len(zi2)):
            if zi2[i] > zi[yID,i,nlays-1]+0.5:
                mgzi2.mask[i,j] = True
            if zi2[i] < zi[yID,i,0]-0.5:
                mgzi2.mask[i,j] = True

    return mgzi2,zi2

def oldgetYmgz(Ysec,zi,yi,mzi,nlays,xID,mgzExtent):

    yy = np.zeros(mzi.shape[0]*mzi.shape[2])
    zz = np.zeros(mzi.shape[0]*mzi.shape[2])
    mgzz = np.zeros(mzi.shape[0]*mzi.shape[2])
    k = 0
    for l in range(0,nlays):
        for i in range(mzi.shape[0]):
            yy[k] = yi[i,xID]
            zz[k] = zi[i,xID,l]
            mgzz[k] = mzi[i,xID,l]
            k += 1

    # define grid.
    zi2 = np.linspace(mgzExtent[0],mgzExtent[1],mgzExtent[2])

    # grid the data.
    mgzi2 = griddata(yy, zz, mgzz, Ysec, zi2, interp='linear')
    # Mask points above top surface
    for j in range(len(Ysec)):
        for i in range(len(zi2)):
            if zi2[i] > zi[j,xID,nlays-1]+0.5:
                mgzi2.mask[i,j] = True
            if zi2[i] < zi[j,xID,0]-0.5:
                mgzi2.mask[i,j] = True

    return mgzi2,zi2


def getYmgz(Ysec,zi,yi,mzi,nlays,xID,mgzExtent):

    yy = np.zeros(mzi.shape[0]*mzi.shape[2])
    zz = np.zeros(mzi.shape[0]*mzi.shape[2])
    mgzz = np.zeros(mzi.shape[0]*mzi.shape[2])
    k = 0
    for l in range(0,nlays):
        for i in range(mzi.shape[0]):
            yy[k] = yi[i,xID]
            zz[k] = zi[i,xID,l]
            mgzz[k] = mzi[i,xID,l]
            k += 1

    # define grid.
    zi2 = np.linspace(mgzExtent[0],mgzExtent[1],mgzExtent[2])
 
    # grid the data.
    mgzi2 = scipygrid((yy, zz), mgzz, (Ysec[None,:], zi2[:,None]), method='nearest')
    mask = np.zeros((len(zi2),len(Ysec)),dtype=int)
    # Mask points above top surface
    for j in range(len(Ysec)):
        for i in range(len(zi2)):
            if zi2[i] > zi[j,xID,nlays-1]+0.5:
                mask[i,j] = 1
            if zi2[i] < zi[j,xID,0]-0.5:
                mask[i,j] = 1

    mmgzi2 = ma.masked_array(mgzi2, mask=mask)          
    
    return mmgzi2,zi2

def plotXtime(figS,Xsec,zi,yID,base,minBase,sl,nlays,layplot,xlim,ylim):

    # Define the figure size and font
    fig = plt.figure(figsize = (figS[0],figS[1]))
    plt.rc("font", size=10)

    # Fill the space between the top surface and the sea-level in blue
    plt.fill_between(Xsec, zi[yID,:,nlays-1], sl, where=sl > zi[yID,:,nlays-1],
                     color=[0.7,0.9,1.0], alpha='0.5')
    plt.fill_between(Xsec, zi[yID,:,0], minBase, where=minBase < zi[yID,:,0],
                     color=[1.0,0.9,0.6], alpha='0.5')

    # Loop through the layers and plot only every layplot layers
    layID = []
    p = 0
    for i in range(0,nlays,layplot):
        layID.append(i)
        plt.plot(Xsec,zi[yID,:,i],'-',color='k',linewidth=1)
        if i>0:
            plt.plot(Xsec,zi[yID,:,i],'-',color='k',linewidth=1)
        if len(layID) > 1:
            if len(layID)%2 == 0:
                plt.fill_between(Xsec, zi[yID,:,layID[p-1]], zi[yID,:,layID[p]],
                                 color=[255./255.,204./255.,204./255.], alpha='0.7')
            else:
                plt.fill_between(Xsec, zi[yID,:,layID[p-1]], zi[yID,:,layID[p]],
                                 color=[204./255.,204./255.,255./255.], alpha='0.7')
        p += 1

    # Define extent, axes and legend
    plt.xlim( xlim[0], xlim[1] )
    plt.ylim( ylim[0], ylim[1] )
    plt.grid(True)
    plt.title('Stratigraphic layers X cross-section',fontsize=12)
    plt.xlabel('X axis (m)',fontsize=10)
    plt.ylabel('Elevation (m)',fontsize=10)

    return

def plotYtime(figS,Ysec,zi,xID,base,minBase,sl,nlays,layplot,xlim,ylim):

    # Define the figure size and font
    fig = plt.figure(figsize = (figS[0],figS[1]))
    plt.rc("font", size=10)

    # Fill the space between the top surface and the sea-level in blue
    plt.fill_between(Ysec, zi[:,xID,nlays-1], sl, where=sl > zi[:,xID,nlays-1],
                     color=[0.7,0.9,1.0], alpha='0.5')
    plt.fill_between(Ysec, zi[:,xID,0], minBase, where=minBase < zi[:,xID,0],
                     color=[1.0,0.9,0.6], alpha='0.5')

    # Loop through the layers and plot only every layplot layers
    layID = []
    p = 0
    for i in range(0,nlays,layplot):
        layID.append(i)
        plt.plot(Ysec,zi[:,xID,i],'-',color='k',linewidth=1)
        if i>0:
            plt.plot(Ysec,zi[:,xID,i],'-',color='k',linewidth=1)
        if len(layID) > 1:
            if len(layID)%2 == 0:
                plt.fill_between(Ysec, zi[:,xID,layID[p-1]], zi[:,xID,layID[p]],
                                 color=[255./255.,204./255.,204./255.], alpha='0.7')
            else:
                plt.fill_between(Ysec, zi[:,xID,layID[p-1]], zi[:,xID,layID[p]],
                                 color=[204./255.,204./255.,255./255.], alpha='0.7')
        p += 1

    # Define extent, axes and legend
    plt.xlim( xlim[0], xlim[1] )
    plt.ylim( ylim[0], ylim[1] )
    plt.grid(True)
    plt.title('Stratigraphic layers Y cross-section',fontsize=12)
    plt.xlabel('Y axis (m)',fontsize=10)
    plt.ylabel('Elevation (m)',fontsize=10)

    return

def plotXmgz(figS,Xsec,zi,zX,mgzX,nlays,yID,extentX,extentY,minBase,sl):

    fig = plt.figure(figsize = (figS[0],figS[1]))
    plt.rc("font", size=10)

    cm = plt.cm.get_cmap('Oranges')

    plt.plot(Xsec,zi[yID,:,0],'-',color='k',linewidth=3, alpha=0.6)
    plt.plot(Xsec,zi[yID,:,nlays-1],'-',color='k',linewidth=3, alpha=0.6)

    # Contour the gridded data, plotting dots at the nonuniform data points.
    CS = plt.contour(Xsec, zX, mgzX, 15, linewidths=0.5, colors='k')
    CS = plt.contourf(Xsec, zX, mgzX, 15, cmap=cm,
                      vmax=abs(mgzX).max(), vmin=mgzX.min())

    cb = plt.colorbar()
    cb.set_label('Mean Grain Size (mm)',fontsize=12)

    # Fill the space between the top surface and the sea-level in blue
    plt.fill_between(Xsec, zi[yID,:,nlays-1], sl, where=sl > zi[yID,:,nlays-1], color=[0.7,0.9,1.0], alpha='0.5')
    plt.fill_between(Xsec, zi[yID,:,0], minBase, where= minBase < zi[yID,:,0], color=[1.0,0.9,0.6], alpha='0.5')

    plt.xlim(extentX[0],extentX[1])
    plt.ylim(extentY[0],extentY[1])
    plt.grid(True)
    plt.title('Y cross-section stratigraphic heterogeneities',fontsize=12)
    plt.xlabel('Y axis (m)',fontsize=10)
    plt.ylabel('Elevation (m)',fontsize=10)

    plt.show()

    return

def plotYmgz(figS,Ysec,zi,zY,mgzY,nlays,xID,extentX,extentY,minBase,sl):

    fig = plt.figure(figsize = (figS[0],figS[1]))
    plt.rc("font", size=10)

    cm = plt.cm.get_cmap('Oranges')

    plt.plot(Ysec,zi[:,xID,0],'-',color='k',linewidth=3, alpha=0.6)
    plt.plot(Ysec,zi[:,xID,nlays-1],'-',color='k',linewidth=3, alpha=0.6)

    # Contour the gridded data, plotting dots at the nonuniform data points.
    CS = plt.contour(Ysec, zY, mgzY, 15, linewidths=0.5, colors='k')
    CS = plt.contourf(Ysec, zY, mgzY, 15, cmap=cm,
                      vmax=abs(mgzY).max(), vmin=mgzY.min())

    cb = plt.colorbar()
    cb.set_label('Mean Grain Size (mm)',fontsize=12)

    # Fill the space between the top surface and the sea-level in blue
    plt.fill_between(Ysec, zi[:,xID,nlays-1], sl, where=sl > zi[:,xID,nlays-1], color=[0.7,0.9,1.0], alpha='0.5')
    plt.fill_between(Ysec, zi[:,xID,0], minBase, where= minBase < zi[:,xID,0], color=[1.0,0.9,0.6], alpha='0.5')

    plt.xlim(extentX[0],extentX[1])
    plt.ylim(extentY[0],extentY[1])
    plt.grid(True)
    plt.title('Y cross-section stratigraphic heterogeneities',fontsize=12)
    plt.xlabel('Y axis (m)',fontsize=10)
    plt.ylabel('Elevation (m)',fontsize=10)

    plt.show()

    return

def plotYmgzReef(figS,Ysec,zi,zY,mgzY,nlays,xID,extentX,extentY,minBase,sl,layplot):

    fig = plt.figure(figsize = (figS[0],figS[1]))
    plt.rc("font", size=10)

    cm = plt.cm.get_cmap('hsv')
    norm = matplotlib.colors.Normalize(clip=True,vmin=0.2, vmax=0.3)
    m = plt.cm.ScalarMappable(cmap=cm,norm=norm)
    m.set_array(mgzY)
    cb = plt.colorbar(m)
    cb.set_label('Mean Grain Size (mm)',fontsize=12)

    plt.plot(Ysec,zi[:,xID,0],'-',color='k',linewidth=3, alpha=0.6)
    plt.plot(Ysec,zi[:,xID,nlays-1],'-',color='k',linewidth=3, alpha=0.6)

    # Contour the gridded data, plotting dots at the nonuniform data points.
    CS = plt.contour(Ysec, zY, mgzY, 25, linewidths=0.5, colors='k')
    CS = plt.contourf(Ysec, zY, mgzY, 25, cmap=cm, norm=norm)

    # Fill the space between the top surface and the sea-level in blue
    plt.fill_between(Ysec, zi[:,xID,nlays-1], sl, where=sl > zi[:,xID,nlays-1], color=[0.7,0.9,1.0], alpha='0.5')
    plt.fill_between(Ysec, zi[:,xID,0], minBase, where= minBase < zi[:,xID,0], color=[1.0,0.9,0.6], alpha='0.5')

    # Loop through the layers and plot only every layplot layers
    layID = []
    p = 0
    for i in range(0,nlays,layplot):
        layID.append(i)
        plt.plot(Ysec,zi[:,xID,i],'--',color='k',linewidth=0.25)
        if i>0:
            plt.plot(Ysec,zi[:,xID,i],'--',color='k',linewidth=0.25)

    plt.xlim(extentX[0],extentX[1])
    plt.ylim(extentY[0],extentY[1])
    plt.grid(True)
    plt.title('Y cross-section stratigraphic heterogeneities',fontsize=12)
    plt.xlabel('Y axis (m)',fontsize=10)
    plt.ylabel('Elevation (m)',fontsize=10)

    plt.show()

    return
