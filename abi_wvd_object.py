#!/home/users/wkjones/miniconda2/envs/NEW/bin/python2.7
from __future__ import print_function, division
import numpy as np
from numpy import ma
import xarray as xr
from glob import glob
from datetime import datetime, timedelta
from scipy.ndimage import label as label_features, find_objects
from scipy.signal import convolve
from skimage.morphology import watershed
from dateutil.parser import parse as parse_date
import argparse
from pyproj import Proj

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

def get_abi_pixel_area(dataset):
    lat, lon = get_abi_lat_lon(dataset)
    nadir_res = float(dataset.spatial_resolution.split('km')[0])
    xx, yy = np.meshgrid(dataset.x.data, dataset.y.data)
    lx_factor = np.cos(np.abs(np.radians(dataset.goes_imager_projection.longitude_of_projection_origin-lon))+np.abs(xx))
    ly_factor = np.cos(np.abs(np.radians(dataset.goes_imager_projection.latitude_of_projection_origin-lat))+np.abs(yy))
    area = nadir_res**2/(lx_factor*ly_factor)
    return area

def get_abi_IR(dataset):
    planck_shape = dataset.planck_fk2.shape
    planck_ndims = len(planck_shape)
    data_shape = dataset.Rad.shape
    data_ndims = len(data_shape)
    planck_reshape = planck_shape + (1,)*(data_ndims-planck_ndims)
    bt = (dataset.planck_fk2.data.reshape(planck_reshape) / (np.log((dataset.planck_fk1.data.reshape(planck_reshape) / dataset.Rad.data) + 1)) - dataset.planck_bc1.data.reshape(planck_reshape)) / dataset.planck_bc2.data.reshape(planck_reshape)
    DQF = dataset.DQF.data
    bt[DQF<0] = np.nan
    bt[DQF>1] = np.nan
    return bt

def get_BT_stats(input_files, slices, features):
    i_slices = 2-len(slices)
    if len(input_files) > features.shape[0]:
        files = input_files[slices[0]]
    else:
        files = input_files
    BT_array = ma.array(np.empty(features.shape), mask=np.logical_not(features))
    for i,f in enumerate(files):
        with xr.open_dataset(f) as C14:
            BT_array.data[i] = get_abi_IR(C14.isel(x=slices[i_slices+1], y=slices[i_slices]))
    return(BT_array)

def get_object_BT(object_info, lat, lon, pixel_area):
    object_info['central_latlon'] = [(np.nanmean(lat[object_info['slice']][mask]),np.nanmean(lon[object_info['slice']][mask])) for mask in object_info['feature_mask']]
    object_info['local_time'] = [f[0]+timedelta(hours=(object_info['central_latlon'][i][1]/15)) for i,f in enumerate(object_info['files'])]
    C13_files = [f[13] for f in object_info['files']]
    object_info['pixel_count'] = [np.sum(mask) for mask in object_info['feature_mask']]
    object_info['area'] = [np.sum(pixel_area[object_info['slice']][mask]) for mask in object_info['feature_mask']]
    print(np.nanmin(object_info['area']))
    object_info['timedeltas'] = [(object_info['files'][i][0]-object_info['files'][i-1][0]).total_seconds() for i in range(1, len(object_info['files']))]
    object_info['growth_rate'] = [(object_info['area'][i]-object_info['area'][i-1])/object_info['timedeltas'][i-1] for i in range(1, len(object_info['files']))]
    object_info['BT'] = get_BT_stats([f[13] for f in object_info['files']], object_info['slice'], object_info['feature_mask'])
    object_info['peak_BT'] = np.nanmin(object_info['BT'], axis=(1,2))
    object_info['cooling_rate'] = [-(object_info['peak_BT'][i]-object_info['peak_BT'][i-1])/object_info['timedeltas'][i-1] for i in range(1, len(object_info['peak_BT']))]
    return object_info

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


features_path = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES16/features/'
features_files = glob(features_path+'abi_RadC_features*2018-06-19*.npz')
features_files.sort()

file_list = []
for f in features_files[2:-1]:
    print(f)
    with np.load(f, allow_pickle=True) as ff:
        files = ff['files']
        file_list.extend(files)

file_set = set([tuple(f) for f in file_list])
filenames = [list(f) for f in sorted(file_set)]
filename_times = [f[0] for f in filenames]

big_mask = np.zeros((len(filenames),1500,2500))
for f in features_files[1:]:
    print(f)
    with np.load(f, allow_pickle=True) as ff:
        files = ff['files']
        labels = ff['labels']
    file_times = [f[0] for f in files]
    where_files = [f in filename_times for f in file_times]
    where_filenames = [f in file_times for f in filename_times]
    big_mask[where_filenames] += labels[where_files]

big_mask = big_mask>0
big_labels = label_features(big_mask)
big_slices = find_objects(big_labels[0])

with xr.open_dataset(file_list[0][13]) as ds:
    lat, lon = get_abi_lat_lon(ds)
    area = get_abi_pixel_area(ds)

objects = []
for i,obj in enumerate(big_slices):
     if obj is not None:
        objects.append(get_object_info(obj, filenames, big_labels[0], lat, lon, area, i+1))
del big_labels
#objects = [get_object_BT(obj, lat, lon, area) for obj in objects]
objects = np.array(objects)

np.savez('/home/users/wkjones/abi_objects_test', objects=objects)
