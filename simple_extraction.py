import tifffile as tif
import numpy as np
import matplotlib.pyplot as plt
import zarr
import os
from roipoly import RoiPoly, MultiRoi

pathname = os.path.normpath('Gao/mouse4_1230um_500hz_1445_330_LED_36_preview_xy_demotion.tif')

imageData = tif.imread(pathname,aszarr=True)
imageData = zarr.open(imageData)

avg = np.average(imageData,axis=0)
print(imageData.shape)

def roi_poly(avg, num):
    n,m = avg.shape
    ROIs = np.zeros((num,n,m))
    plt.imshow(avg)
    multiroi_named = MultiRoi()
    i = 0
    for name, roi in multiroi_named.rois.items():
        mask = roi.get_mask(avg)
        ROIs[i] = mask
        i = i+1
        
    return ROIs

rois = roi_poly(avg, 5)

def context_region(clnmsk, pix_pad=0):
    n,m = clnmsk.shape
    rows = np.any(clnmsk, axis=1)
    cols = np.any(clnmsk, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    top = rmin-pix_pad
    if top < 0:
        top = 0
    bot = rmax+pix_pad
    if bot > n:
        bot = n
    lef = cmin-pix_pad
    if lef < 0:
        lef = 0
    rig = cmax+pix_pad
    if rig > m:
        rig = m
    
    return top,bot,lef,rig


masks = [object]*5
traces = np.zeros((5,len(imageData)))
fig, ax = plt.subplots(5,2)

for i in range(5):
    masks_coords = context_region(rois[i], 0)

    temp_mask = rois[i][masks_coords[0]:masks_coords[1],masks_coords[2]:masks_coords[3]]
    context_roi = imageData[:,masks_coords[0]:masks_coords[1],masks_coords[2]:masks_coords[3]]

    trace = context_roi*temp_mask[np.newaxis,:,:]
    trace = np.average(trace,axis=(1,2))
    traces[i] = trace
    masks[i] = temp_mask

    ax[i,0].imshow(temp_mask)
    ax[i,1].plot(trace)

plt.show()