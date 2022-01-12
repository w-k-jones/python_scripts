#!/home/users/wkjones/miniconda2/envs/slot_process/bin/python
import os
from glob import glob
from google.cloud import storage
import tarfile
import argparse

import numpy as np
from numpy import ma
import pandas as pd
import xarray as xr
import dask.array as da
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from pyproj import Proj, Geod
from scipy import ndimage as ndi
from python_toolbox import abi_tools, dataset_tools, opt_flow, slots
import cv2 as cv

parser = argparse.ArgumentParser(description="""Detect deep convective features
    in GOES-16 ABI imagery using a semi-lagrangian method. Using preprocessed
    data""")
parser.add_argument('satellite', help='GOES satellite to use (16 or 17)', type=int)
parser.add_argument('start_date', help='Start date of processing', type=str)
parser.add_argument('end_date', help='End date of processing', type=str)
parser.add_argument('-C', help='Use CONUS scan (5 minute) data', action='store_true', default=False)
parser.add_argument('-F', help='Use Full scan (15 minute) data', action='store_true', default=False)
parser.add_argument('-l', help='Subset length scale', default=1, type=int)
parser.add_argument('-x0', help='Initial subset x location', default=0, type=int)
parser.add_argument('-x1', help='End subset x location', default=2500, type=int)
parser.add_argument('-y0', help='Initial subset y location', default=0, type=int)
parser.add_argument('-y1', help='End subset y location', default=2500, type=int)

args = parser.parse_args()
satellite = str(args.satellite)
start_date = parse_date(args.start_date)
end_date = parse_date(args.end_date)
l = int(args.l)
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

goes_dir = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES16/MCMIPC/'
save_dir = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES'+satellite+'/features_downscale/'+str(start_date.year).zfill(4)+'/'

if not os.path.isdir(goes_dir):
    os.makedirs(goes_dir)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

def get_goes_MCMIPC(date, save_dir='./', n_pad=0):
    storage_client = storage.Client()
    goes_bucket = storage_client.get_bucket('gcp-public-data-goes-16')

    s_year = str(date.year).zfill(4)
    doy = (date - datetime(date.year,1,1)).days+1
    s_doy = str(doy).zfill(3)
    s_hour = str(date.hour).zfill(2)

    save_path = save_dir + '/' + s_year + '/' + s_doy + '/' + s_hour + '/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    files = [f.split('/')[-1] for f in
             glob(save_path + 'OR_ABI-L2-MCMIPC-M[36]_G16_s'
                  + s_year + s_doy + s_hour + '*.nc')]

    blobs = goes_bucket.list_blobs(
            prefix='ABI-L2-MCMIPC/' + s_year + '/' + s_doy + '/'+s_hour \
                   + '/OR_ABI-L2-MCMIPC-',
            delimiter='/'
            )

    for blob in blobs:
        if blob.name.split('/')[-1] not in files:
            print(blob.name.split('/')[-1])
            blob.download_to_filename(save_path + blob.name.split('/')[-1])

    goes_files = glob(save_path + '/OR_ABI-L2-MCMIPC-M[36]_G16_s'
                      + s_year + s_doy + s_hour + '*.nc')

    if n_pad>0:
        date_pre = date - timedelta(hours=1)
        s_year = str(date_pre.year).zfill(4)
        doy = (date_pre - datetime(date_pre.year,1,1)).days+1
        s_doy = str(doy).zfill(3)
        s_hour = str(date_pre.hour).zfill(2)
        save_path = save_dir + '/' + s_year + '/' + s_doy + '/' + s_hour + '/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        files = [f.split('/')[-1] for f in
                 glob(save_path + '/OR_ABI-L2-MCMIPC-M[36]_G16_s'+s_year+s_doy+s_hour+'*.nc')]
        blobs = list(goes_bucket.list_blobs(
                    prefix='ABI-L2-MCMIPC/'+s_year+'/'+s_doy+'/'+s_hour+'/OR_ABI-L2-MCMIPC-',
                    delimiter='/'
                    ))[-n_pad:]

        for blob in blobs:
            if blob.name.split('/')[-1] not in files:
                print(blob.name.split('/')[-1])
                blob.download_to_filename(save_path + blob.name.split('/')[-1])

        goes_files = glob(save_path + '/OR_ABI-L2-MCMIPC-M[36]_G16_s'+s_year+s_doy+s_hour+'*.nc')[-n_pad:] + goes_files

        date_next = date + timedelta(hours=1)
        s_year = str(date_next.year).zfill(4)
        doy = (date_next - datetime(date_pre.year,1,1)).days+1
        s_doy = str(doy).zfill(3)
        s_hour = str(date_next.hour).zfill(2)
        save_path = save_dir + '/' + s_year + '/' + s_doy + '/' + s_hour + '/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        files = [f.split('/')[-1] for f in
                 glob(save_path + '/OR_ABI-L2-MCMIPC-M[36]_G16_s'+s_year+s_doy+s_hour+'*.nc')]
        blobs = list(goes_bucket.list_blobs(
                    prefix='ABI-L2-MCMIPC/'+s_year+'/'+s_doy+'/'+s_hour+'/OR_ABI-L2-MCMIPC-',
                    delimiter='/'
                    ))[:n_pad]

        for blob in blobs:
            if blob.name.split('/')[-1] not in files:
                print(blob.name.split('/')[-1])
                blob.download_to_filename(save_path + blob.name.split('/')[-1])

        goes_files += glob(save_path + '/OR_ABI-L2-MCMIPC-M[36]_G16_s'+s_year+s_doy+s_hour+'*.nc')[:n_pad]

    return goes_files

goes_files = set(
                sum([sorted(get_goes_MCMIPC(date, save_dir=goes_dir, n_pad=1))
                  for date in
                  pd.date_range(start_date, end_date, freq='H', closed='left').to_pydatetime().tolist()], []))

goes_files = sorted(list(goes_files))

file_iter = goes_files.copy()

for file in file_iter:
    try:
        temp = xr.open_mfdataset(file, concat_dim='t', combine='nested')
    except OSError:
        print('Unable to open file:', file)
        goes_files.remove(file)
    else:
        temp.close()

print('Discovered MCMIPC files:', len(goes_files))

file_slice = {'y':slice(y0, y1), 'x':slice(x0, x1)}
try:
    goes_ds = xr.open_mfdataset(goes_files, concat_dim='t', combine='nested').isel(file_slice)
                                # parallel=True).isel(file_slice)
except OSError:
    for file in file_iter:
        try:
            temp = xr.open_mfdataset(file, concat_dim='t', combine='nested')
        except OSError:
            print('Unable to open file:', file)
            goes_files.remove(file)
        else:
            temp.close()
    goes_ds = xr.open_mfdataset(goes_files, concat_dim='t', combine='nested').isel(file_slice)
                                # parallel=True).isel(file_slice)

goes_dates = [abi_tools.get_abi_date_from_filename(f) for f in goes_files]

print(datetime.now(),'Loading data from channel 8')
C8_data = goes_ds.CMI_C08.astype(np.float32).data.compute()
print(datetime.now(),'Loading data from channel 10')
C10_data = goes_ds.CMI_C10.astype(np.float32).data.compute()
print(datetime.now(),'Loading data from channel 13')
C13_data = goes_ds.CMI_C13.astype(np.float32).data.compute()
print(datetime.now(),'Loading data from channel 15')
C15_data = goes_ds.CMI_C15.astype(np.float32).data.compute()

wvd = (C8_data - C10_data)
swd = (C13_data - C15_data)

goes_ds.close()

out_dims = goes_ds.CMI_C13.dims
out_coords = goes_ds.CMI_C13.coords

del C8_data
del C10_data
del C15_data

print(datetime.now(), 'Predicting flow')
flow_kwargs = {'pyr_scale':0.5, 'levels':7, 'winsize':64, 'iterations':4,
               'poly_n':7, 'poly_sigma':1.5,
               'flags':cv.OPTFLOW_FARNEBACK_GAUSSIAN}
flow = slots.get_flow_func(C13_data, post_iter=3, compute=True,
                           **flow_kwargs, dtype=np.float32)

print(datetime.now(), 'Calculating Growth Metric')
dt = [(goes_dates[1]-goes_dates[0]).total_seconds()/60] \
     + [(goes_dates[i+2]-goes_dates[i]).total_seconds()/120 \
        for i in range(len(goes_files)-2)] \
     + [(goes_dates[-1]-goes_dates[-2]).total_seconds()/60]
dt = np.array(dt)

bt_dt = slots.get_flow_diff(C13_data, flow, dtype=np.float32) / dt[:,np.newaxis,np.newaxis]

filter_thresh = 5
filter_range = 11
delta_bt = C13_data - ndi.morphology.grey_erosion(C13_data,
                               (1, filter_range, filter_range)).astype(np.float32)
bt_filter = np.maximum((filter_thresh-delta_bt)/filter_thresh,0).astype(np.float32)

bt_growth = (np.maximum(-bt_dt,0) * bt_filter).astype(np.float32)

del dt, bt_dt, delta_bt, bt_filter, C13_data


bt_growth = da.nanmean(slots.flow_convolve(bt_growth, flow,
                       structure=ndi.generate_binary_structure(3,1)), 0).compute()

print(datetime.now(), 'Downscaling variables')
out_coords = {'t':out_coords['t'],
              'y':dataset_tools.ds_area_func(np.nanmean, out_coords['y'], l),
              'x':dataset_tools.ds_area_func(np.nanmean, out_coords['x'], l)}

bt_growth = dataset_tools.apply_area_func(np.nanmax, bt_growth, l, axis=(1,2))

wvd = dataset_tools.apply_area_func(np.nanmax, wvd, l, axis=(1,2))
swd = dataset_tools.apply_area_func(np.nanmax, swd, l, axis=(1,2))

flow = slots.Flow_Func(dataset_tools.apply_area_func(
                           np.nanmean, flow.flow_x_for, l, axis=(1,2)),
                       dataset_tools.apply_area_func(
                           np.nanmean, flow.flow_x_back, l, axis=(1,2)),
                       dataset_tools.apply_area_func(
                           np.nanmean, flow.flow_y_for, l, axis=(1,2)),
                       dataset_tools.apply_area_func(
                           np.nanmean, flow.flow_y_back, l, axis=(1,2)))

print(datetime.now(), 'Calculating markers and mask')
upper_thresh = -5
lower_thresh = -15
growth_thresh = 0.5
markers = np.logical_and(wvd>upper_thresh,  bt_growth>growth_thresh)
markers = markers.astype('bool')

mask = (wvd<lower_thresh)
mask = ndi.grey_erosion(mask, (np.minimum(5, (mask.shape[0]-1)//2), 5, 5)).astype('bool')
mask = np.logical_or(mask, np.isnan(wvd))

print(datetime.now(), 'Calulcating inner edges')
inner_edges = slots.flow_sobel(np.minimum(np.maximum(
                               wvd - swd + (bt_growth * 5),
                               lower_thresh), upper_thresh),
                        flow, direction='uphill', magnitude=True,
                        dtype=np.float32).compute()

print(datetime.now(), 'Watershedding inner region')

inner = slots.watershed(inner_edges, flow,
                        markers, mask=mask,
                        structure=ndi.generate_binary_structure(3,1)
                        ).astype(bool)

del inner_edges

print(datetime.now(), 'Calulcating outer edges')
outer_edges = slots.flow_sobel(np.minimum(np.maximum(
                               wvd + swd + (bt_growth * 5) - 5,
                               lower_thresh + 5), upper_thresh),
                        flow, direction='uphill', magnitude=True,
                        dtype=np.float32).compute()


print(datetime.now(), 'Watershedding outer region')

outer = slots.watershed(outer_edges, flow,
                        markers, mask=mask,
                        structure=ndi.generate_binary_structure(3,1)
                        ).astype(bool)

del outer_edges

ds = xr.Dataset({'inner':xr.DataArray(inner.astype(bool), dims=out_dims, coords=out_coords),
                 'outer':xr.DataArray(outer.astype(bool), dims=out_dims, coords=out_coords)})

save_name = 'abi_Rad'+scan_type+'_G'+satellite+'_flow_features_S'\
            +start_date.isoformat()+'_E'+end_date.isoformat()\
            +'_x0'+str(x0).zfill(4)+'_x1'+str(x1).zfill(4)\
            +'_y0'+str(y0).zfill(4)+'_y1'+str(y1).zfill(4)

print(datetime.now(), 'Saving file to:')
print(save_dir+save_name+'.nc')
ds.to_netcdf(save_dir+save_name+'.nc', format='NETCDF4')

print(datetime.now(), 'Feature detection successfully completed')
