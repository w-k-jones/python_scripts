#!/home/users/wkjones/miniconda2/envs/slot_process/bin/python
import os
from glob import glob
import argparse
import gc

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from python_toolbox import slots
from scipy import ndimage as ndi

parser = argparse.ArgumentParser(description="""Detect deep convective features
    in GOES-16 ABI imagery using a semi-lagrangian method. Using preprocessed
    data""")
parser.add_argument('satellite', help='GOES satellite to use (16 or 17)', type=int)
parser.add_argument('start_date', help='Start date of processing', type=str)
parser.add_argument('end_date', help='End date of processing', type=str)
parser.add_argument('-C', help='Use CONUS scan (5 minute) data', action='store_true', default=False)
parser.add_argument('-F', help='Use Full scan (15 minute) data', action='store_true', default=False)
parser.add_argument('-x0', help='Initial subset x location', default=0, type=int)
parser.add_argument('-x1', help='End subset x location', default=2500, type=int)
parser.add_argument('-y0', help='Initial subset y location', default=0, type=int)
parser.add_argument('-y1', help='End subset y location', default=2500, type=int)

#parser.add_argument('-l', help='Contraction length scale: takes the max value over l*l pixels', default=1)

args = parser.parse_args()
satellite = str(args.satellite)
start_date = parse_date(args.start_date)
end_date = parse_date(args.end_date)
x0 = int(args.x0)
x1 = int(args.x1)
y0 = int(args.y0)
y1 = int(args.y1)

if args.C:
    scan_type = 'C'
elif args.F:
    scan_type = 'F'

print(datetime.now(),'Detecting features for satellite GOES'+satellite)
print('Start date:', start_date.isoformat())
print('End date:', end_date.isoformat())
date_list = pd.date_range(start_date, end_date, freq='H', closed='left').to_pydatetime().tolist()

preproc_dir = '/work/scratch-nompiio/wkjones/preprocessing/'

preproc_files = []
for date in date_list:
    s_date = date.isoformat()
    preproc_files.extend(glob(preproc_dir + '*' + s_date + '.nc'))

print('Discovered preprocess files:', len(preproc_files))

file_slice = {'y':slice(y0, y1), 'x':slice(x0, x1)}
preproc_ds = xr.open_mfdataset(preproc_files, concat_dim='t', combine='nested',
                               parallel=True).isel(file_slice)
out_dims = ('t','y','x')
out_coords = preproc_ds.coords

print('Loading flow func', datetime.now())

flow = slots.Flow_Func(preproc_ds.flow_x_for.data.compute(),
                       preproc_ds.flow_x_back.data.compute(),
                       preproc_ds.flow_y_for.data.compute(),
                       preproc_ds.flow_y_back.data.compute())

print('Loading markers', datetime.now())

markers = preproc_ds.markers.data.compute()

print('Loading mask', datetime.now())

mask = preproc_ds.mask.data.compute()

print('Loading inner edges', datetime.now())

edges = preproc_ds.inner_edges.data.compute()

print('Watershed inner region', datetime.now())

inner = slots.watershed(edges, flow, markers, mask=mask).astype(bool)

del edges
gc.collect()

print('Loading outer edges', datetime.now())

edges = preproc_ds.outer_edges.data.compute()

print('Watershed outer region', datetime.now())

new_markers = np.logical_or(
    ndi.grey_erosion(inner, size=(1,3,3)).astype(bool),
    markers)

outer = slots.watershed(edges, flow, new_markers, mask=mask).astype(bool)

ds = xr.Dataset({'inner':xr.DataArray(inner.astype(bool), dims=out_dims, coords=out_coords),
                 'outer':xr.DataArray(outer.astype(bool), dims=out_dims, coords=out_coords)})

save_name = 'abi_Rad'+scan_type+'_G'+satellite+'_flow_features_S'\
            +start_date.isoformat()+'_E'+end_date.isoformat()\
            +'_x0'+str(x0).zfill(4)+'_x1'+str(x1).zfill(4)\
            +'_y0'+str(y0).zfill(4)+'_y1'+str(y1).zfill(4)
save_dir = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES'+satellite+'/features/'+str(start_date.year).zfill(4)+'/'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

print(datetime.now(), 'Saving file to:')
print(save_dir+save_name+'.nc')
ds.to_netcdf(save_dir+save_name+'.nc', format='NETCDF4')

print(datetime.now(), 'Feature detection successfully completed')
