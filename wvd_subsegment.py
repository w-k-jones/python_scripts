#!/home/users/wkjones/miniconda2/envs/python3/bin/python2.7
from __future__ import print_function, division
import numpy as np
from numpy import ma
import xarray as xr
from glob import glob
from datetime import datetime, timedelta
from scipy.ndimage import label as label_features, find_objects
from scipy.signal import convolve
from skimage.feature import peak_local_max
from skimage.morphology import local_minima, h_minima, selem, star, ball, watershed
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import grey_erosion, grey_dilation, grey_opening, grey_closing, binary_opening, binary_dilation
import pandas as pd
from pyproj import Proj
import tobac

def get_abi_IR(dataset, check=False):
    bt = (dataset.planck_fk2 / (np.log((dataset.planck_fk1 / dataset.Rad) + 1)) - dataset.planck_bc1) / dataset.planck_bc2
    if check:
        DQF = dataset.DQF
        bt[DQF<0] = np.nan
        bt[DQF>1] = np.nan
    return bt

def get_abi_lat_lon(dataset):
    p = Proj(proj='geos', h=dataset.goes_imager_projection.perspective_point_height,
             lon_0=dataset.goes_imager_projection.longitude_of_projection_origin,
             sweep=dataset.goes_imager_projection.sweep_angle_axis)
    xx, yy = np.meshgrid(dataset.x.data*dataset.goes_imager_projection.perspective_point_height,
                         dataset.y.data*dataset.goes_imager_projection.perspective_point_height)
    lons, lats = p(xx, yy, inverse=True)
    lons[lons>=1E30] = np.nan
    lats[lats>=1E30] = np.nan
    return lats, lons

def subsegment(bt, h_level=2.5, min_separation=2):
    bt.fill_value = bt.max()
    peaks = np.all((h_minima(gaussian_filter(bt.filled(),0.5),h_level),local_minima(gaussian_filter(bt.filled(),0.5),connectivity=min_separation),bt.filled()<240), axis=0)
    segments = watershed(bt.filled(),label_features(peaks)[0], mask=np.logical_not(bt.mask))
    return segments

def get_central_xy(obj):
    xx,yy = np.meshgrid(np.arange(obj['slice'][1].start,obj['slice'][1].stop),np.arange(obj['slice'][0].start,obj['slice'][0].stop))
    central_x = ma.array(np.stack([xx]*obj['feature_mask'].shape[0]),mask=np.logical_not(obj['feature_mask'])).mean(axis=(1,2)).data
    central_y = ma.array(np.stack([yy]*obj['feature_mask'].shape[0]),mask=np.logical_not(obj['feature_mask'])).mean(axis=(1,2)).data
    return central_x, central_y

def subsegment_object(obj):
    c13_ds = xr.open_mfdataset([f[13] for f in obj['files']], concat_dim='t')
    BT = get_abi_IR(c13_ds[{'y':obj['slice'][0], 'x':obj['slice'][1]}]).compute()
    BTma = BT.to_masked_array()
    BTma.mask = np.logical_not(obj['feature_mask'])
    segment_labels = subsegment(BTma)
    return segment_labels

def get_object_info(object_slice, files_list, feature_mask, lats, lons, pixel_area, label=True):
    object_info={}
    object_info['files']=files_list[object_slice[0]]
    object_info['slice']=object_slice[1:]
    object_info['duration'] = (object_info['files'][-1][0] - object_info['files'][0][0]).total_seconds()
    y0 = object_slice[1].start
    y1 = object_slice[1].stop-1
    x0 = object_slice[2].start
    x1 = object_slice[2].stop-1
    object_info['UL_corner_latlon'] = [lats[y0,x0],lons[y0,x0]]
    object_info['LL_corner_latlon'] = [lats[y1,x0],lons[y1,x0]]
    object_info['UR_corner_latlon'] = [lats[y0,x1],lons[y0,x1]]
    object_info['LR_corner_latlon'] = [lats[y1,x1],lons[y1,x1]]
    object_info['feature_mask'] = feature_mask[object_slice]==label
    object_info['central_latlon'] = [(np.nanmean(lats[object_info['slice']][mask]),np.nanmean(lons[object_info['slice']][mask])) for mask in object_info['feature_mask']]
    object_info['local_time'] = [f[0]+timedelta(hours=(object_info['central_latlon'][i][1]/15)) for i,f in enumerate(object_info['files'])]
    object_info['pixel_count'] = [np.sum(mask) for mask in object_info['feature_mask']]
    object_info['area'] = [np.sum(pixel_area[object_info['slice']][mask]) for mask in object_info['feature_mask']]
    object_info['timedeltas'] = [(object_info['files'][i][0]-object_info['files'][i-1][0]).total_seconds() for i in range(1, len(object_info['files']))]
    object_info['growth_rate'] = [(object_info['area'][i]-object_info['area'][i-1])/object_info['timedeltas'][i-1] for i in range(1, len(object_info['files']))]
    return object_info

frame = []
hdim_1 = []
hdim_2 = []
idx = []
num = []
threshold_value = []
feature = []
time = []
timestr = []
latitude = []
longitude = []

print('Loading file')
with np.load('/home/users/wkjones/abi_objects_test.npz', allow_pickle=True) as objects_file:
    objects = objects_file['objects']

print('Getting lats/lons')
with xr.open_dataset(objects[0]['files'][0][13]) as ds:
    lats, lons = get_abi_lat_lon(ds)

print('Looping over objects')
idx_base = 0
for i, obj in enumerate(objects):
    if np.any([slc.start==0 for slc in obj['slice']]) or np.any([slc.start==[287,1500,2500][j] for j,slc in enumerate(obj['slice'])]):
        pass
    else:
        xx,yy = np.meshgrid(np.arange(obj['slice'][1].start,obj['slice'][1].stop),np.arange(obj['slice'][0].start,obj['slice'][0].stop))
        segments = subsegment_object(obj)
        for j in range(segments.shape[0]):
            for k in range(1,np.max(segments)+1):
                if np.any(segments[j] == k):
                    mask = np.logical_not(segments[j]==k)
                    hdim_1.append(ma.array(xx,mask=mask).mean())
                    hdim_2.append(ma.array(yy,mask=mask).mean())
                    idx.append(idx_base+j)
                    num.append(np.sum(segments[j]==k))
                    threshold_value.append(240)
                    time.append(obj['files'][j][0])
                    latitude.append(ma.array(lats[obj['slice'][0],obj['slice'][1]],mask=mask).mean())
                    longitude.append(ma.array(lons[obj['slice'][0],obj['slice'][1]],mask=mask).mean())
        idx_base += np.max(segments)

time = np.array(time)
hdim_1 = np.array(hdim_1)
hdim_2 = np.array(hdim_2)
idx = np.array(idx)
num = np.array(num)
threshold_value = np.array(threshold_value)
timestr = np.array([str(dt) for dt in time])
latitude = np.array(latitude)
longitude = np.array(longitude)
feature = np.arange(1,time.size+1)
uniq_dates = list(np.unique(time))
date_range = pd.date_range(min(uniq_dates),max(uniq_dates),(max(uniq_dates)-min(uniq_dates)).total_seconds()//300+1).tolist()
frame = np.array([date_range.index(t) for t in time])

feature_df = pd.DataFrame(dict(frame=frame,hdim_1=hdim_1,hdim_2=hdim_2,idx=idx,num=num,threshold_value=threshold_value,feature=feature,time=time,timestr=timestr,latitude=latitude,longitude=longitude))

feature_df.to_pickle('/home/users/wkjones/abi_dataframe_pre_track.pkl')

dt,dxy = 300,2000

parameters_linking={}
parameters_linking['v_max']=30
parameters_linking['stubs']=2
parameters_linking['order']=1
parameters_linking['extrapolate']=2
parameters_linking['memory']=3 #Added 1 step of memory to deal with the missing frames, perhaps more needed?
parameters_linking['adaptive_stop']=0.2
parameters_linking['adaptive_step']=0.95
parameters_linking['subnetwork_size']=100
parameters_linking['method_linking']= 'predict'

print('tracking')
Track=tobac.linking_trackpy(feature_df,np.zeros((1500,2500)),dt=dt,dxy=dxy,**parameters_linking)

Track.to_pickle('/home/users/wkjones/abi_dataframe_track.pkl')
