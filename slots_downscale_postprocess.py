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
from dateutil.relativedelta import relativedelta
from python_toolbox import slots
from scipy import ndimage as ndi

parser = argparse.ArgumentParser(description="""Detect deep convective features
    in GOES-16 ABI imagery using a semi-lagrangian method. Using preprocessed
    data""")
parser.add_argument('satellite', help='GOES satellite to use (16 or 17)', type=int)
parser.add_argument('start_date', help='Start date of processing', type=str)
# parser.add_argument('end_date', help='End date of processing', type=str)
parser.add_argument('-C', help='Use CONUS scan (5 minute) data', action='store_true', default=False)
parser.add_argument('-F', help='Use Full scan (15 minute) data', action='store_true', default=False)
parser.add_argument('-l', help='Subset length scale', default=1, type=int)
parser.add_argument('-x0', help='Initial subset x location', default=0, type=int)
parser.add_argument('-x1', help='End subset x location', default=1500, type=int)
parser.add_argument('-y0', help='Initial subset y location', default=0, type=int)
parser.add_argument('-y1', help='End subset y location', default=2500, type=int)
parser.add_argument('-pd', help='Directory to locate processed files',
                    default='/work/scratch-nompiio/wkjones/processing_downscale/',
                    type=str)
parser.add_argument('-sd', help='Directory to save processed files',
                    default='/work/scratch-nompiio/wkjones/postprocessing_downscale/',
                    type=str)

#parser.add_argument('-l', help='Contraction length scale: takes the max value over l*l pixels', default=1)

args = parser.parse_args()
satellite = str(args.satellite)
start_date = parse_date(args.start_date)
# end_date = parse_date(args.end_date)
l = int(args.l)
x0 = int(args.x0)//l
x1 = int(args.x1)//l
y0 = int(args.y0)//l
y1 = int(args.y1)//l

if args.C:
    scan_type = 'C'
elif args.F:
    scan_type = 'F'

print(datetime.now(),'Detecting features for satellite GOES'+satellite)
print('Start date:', start_date.isoformat())

proc_dir = args.pd+'/'+str(l)+'/'
save_dir = args.sd+'/'+str(l)+'/'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

ds_files = sorted(glob(proc_dir+'slots_process_*'+start_date.isoformat()[:7]+'*.nc'))
print(ds_files)
with xr.open_dataset(ds_files[0]) as ds:
    combine_inner = xr.DataArray(ds.inner.data.astype(bool), dims=('t','y','x'),
                                 coords={'t':ds.t.data.copy().astype('datetime64[s]'),
                                         'y':ds.y, 'x':ds.x})
    combine_inner = combine_inner.loc[combine_inner.t>np.datetime64('2016-01-01')]
    combine_outer = xr.DataArray(ds.outer.data.astype(bool), dims=('t','y','x'),
                                 coords={'t':ds.t.data.copy().astype('datetime64[s]'),
                                         'y':ds.y, 'x':ds.x})
    combine_outer = combine_outer.loc[combine_outer.t>np.datetime64('2016-01-01')]


for file in ds_files[1:]:
    print(file)
    with xr.open_dataset(file) as ds:
        new_inner = xr.DataArray(ds.inner.data.astype(bool), dims=('t','y','x'),
                                 coords={'t':ds.t.data.copy().astype('datetime64[s]'),
                                         'y':combine_inner.y, 'x':combine_inner.x})
        new_inner = new_inner.loc[new_inner.t>np.datetime64('2016-01-01')]
        combine_inner = np.logical_or(*xr.align(combine_inner, new_inner, join='outer',
                                                fill_value=0))
        new_outer = xr.DataArray(ds.outer.data.astype(bool), dims=('t','y','x'),
                                 coords={'t':ds.t.data.copy().astype('datetime64[s]'),
                                         'y':combine_outer.y, 'x':combine_outer.x})
        new_outer = new_outer.loc[new_outer.t>np.datetime64('2016-01-01')]
        combine_outer = np.logical_or(*xr.align(combine_outer, new_outer, join='outer',
                                                fill_value=0))

combine_inner = combine_inner.sel(t=slice(start_date, start_date+relativedelta(months=1)))
combine_outer = combine_outer.sel(t=slice(start_date, start_date+relativedelta(months=1)))

ds = xr.Dataset({'inner':combine_inner,
                 'outer':combine_outer})

save_name = 'slots_postprocess_'+scan_type+'_G'+satellite+'_'+start_date.isoformat()
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

print(datetime.now(), 'Saving file to:')
print(save_dir+save_name+'.nc')
ds.to_netcdf(save_dir+save_name+'.nc', format='NETCDF4')

print(datetime.now(), 'Feature detection successfully completed')
