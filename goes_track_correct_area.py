#!/home/users/wkjones/miniconda2/envs/NEW/bin/python2.7
from __future__ import print_function, division
import numpy as np
from numpy import ma
import xarray as xr
from glob import glob
from datetime import datetime, timedelta, date, time
from pyproj import Proj
import argparse

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

print('Loading objects file')
with np.load(filename) as f:
    objects = f['objects']
    labels = f['labels']

print('Finding lat, lon and pixel areas')
with xr.open_dataset(objects[0]['files'][0][14]) as ds:
    lat, lon = get_abi_lat_lon(ds)
    area = get_abi_pixel_area(ds)

print('Correcting areas')
for object_info in objects:
    object_info['area'] = [np.sum(area[object_info['slice']][mask]) for mask in object_info['feature_mask']]
    object_info['growth_rate'] = [(object_info['area'][i]-object_info['area'][i-1])/object_info['timedeltas'][i-1] for i in range(1, len(object_info['files']))]

print('Saving: '+filename)
np.savez(filename, labels=labels, objects=objects)
