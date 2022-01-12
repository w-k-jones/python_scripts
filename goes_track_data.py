#!/home/users/wkjones/miniconda2/envs/NEW/bin/python2.7
from __future__ import print_function, division
import numpy as np
from numpy import ma
import xarray as xr
from glob import glob
import argparse
from datetime import datetime, timedelta
from pyproj import Proj

parser = argparse.ArgumentParser(description='Detect deep convective features in GOES-16 ABI imagery')
parser.add_argument('filename', help='Object file to reprocess', type=str)
args = parser.parse_args()
filename = glob(args.filename)
if len(filename) == 0:
    raise Exception('Invalid file input')
else:
    filename = filename[0]

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

def get_abi_ref(dataset):
    ref = dataset.Rad.data * dataset.kappa0.data
    DQF = dataset.DQF.data
    ref[DQF<0] = np.nan
    ref[DQF>1] = np.nan
    return ref

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
    C14_files = [f[14] for f in object_info['files']]
    object_info['pixel_count'] = [np.sum(mask) for mask in object_info['feature_mask']]
    object_info['area'] = [np.sum(pixel_area[object_info['slice']][mask]) for mask in object_info['feature_mask']]
    print(np.nanmin(object_info['area']))
    object_info['timedeltas'] = [(object_info['files'][i][0]-object_info['files'][i-1][0]).total_seconds() for i in range(1, len(object_info['files']))]
    object_info['growth_rate'] = [(object_info['area'][i]-object_info['area'][i-1])/object_info['timedeltas'][i-1] for i in range(1, len(object_info['files']))]
    object_info['BT'] = get_BT_stats([f[14] for f in object_info['files']], object_info['slice'], object_info['feature_mask'])
    object_info['peak_BT'] = np.nanmin(object_info['BT'], axis=(1,2))
    object_info['cooling_rate'] = [-(object_info['peak_BT'][i]-object_info['peak_BT'][i-1])/object_info['timedeltas'][i-1] for i in range(1, len(object_info['peak_BT']))]
    return object_info

print('Loading objects file')
with np.load(filename) as f:
    objects = f['objects']
    labels = f['labels']

print('Finding lat, lon and pixel areas')
with xr.open_dataset(objects[0]['files'][0][14]) as ds:
    lat, lon = get_abi_lat_lon(ds)
    area = get_abi_pixel_area(ds)

print('Getting BT info for objects')
new_objects = np.array([get_object_BT(object_info, lat, lon, area) for object_info in objects])

savename = '_'.join(filename.split('_')[:-2])+'_BT'+'_'.join(filename.split('_')[-2:])
print('Saving: '+savename)
np.savez(savename, labels=labels, objects=new_objects)
