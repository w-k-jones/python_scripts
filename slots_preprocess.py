#!/home/users/wkjones/miniconda2/envs/slot_process/bin/python
import os
from glob import glob
from google.cloud import storage
import tarfile
import argparse

import numpy as np
import numpy.ma as ma
from scipy import stats
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import pyart
from pyproj import Proj, Geod
from scipy import ndimage as ndi
from python_toolbox import abi_tools, dataset_tools, opt_flow, slots
import cv2 as cv
import dask.array as da

parser = argparse.ArgumentParser(description="""Preprocessor for feature
    detection in goes-abi imagery""")
parser.add_argument('date', help='Start date of processing', type=str)

args = parser.parse_args()
date = parse_date(args.date)

goes_dir = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES16/MCMIPC/'
save_dir = '/work/scratch-nompiio/wkjones/preprocessing/'

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

goes_files = get_goes_MCMIPC(date, save_dir=goes_dir, n_pad=1)

if len(goes_files) < 3:
    raise OSError("No files discovered")

file_iter = goes_files.copy()

for file in file_iter:
    try:
        temp = xr.open_mfdataset(file, concat_dim='t', combine='nested')
    except OSError:
        print('Unable to open file:', file)
        goes_files.remove(file)
    else:
        temp.close()

try:
    goes_ds = xr.open_mfdataset(goes_files, concat_dim='t', combine='nested', parallel=True)
except OSError:
    for file in file_iter:
        try:
            temp = xr.open_mfdataset(file, concat_dim='t', combine='nested')
        except OSError:
            print('Unable to open file:', file)
            goes_files.remove(file)
        else:
            temp.close()
    goes_ds = xr.open_mfdataset(goes_files, concat_dim='t', combine='nested', parallel=True)

def get_abi_date_from_filename(filename):
    base_string = filename.split('/')[-1].split('_s')[-1]
    date = parse_date(base_string[:4]+'0101'+base_string[7:13]) + timedelta(days=int(base_string[4:7])-1)
    return date

goes_dates = [get_abi_date_from_filename(f) for f in goes_files]

print(datetime.now(),'Loading data from channel 8')
C8_data = goes_ds.CMI_C08.astype(np.float32).compute()
print(datetime.now(),'Loading data from channel 10')
C10_data = goes_ds.CMI_C10.astype(np.float32).compute()
print(datetime.now(),'Loading data from channel 13')
C13_data = goes_ds.CMI_C13.astype(np.float32).compute()
print(datetime.now(),'Loading data from channel 15')
C15_data = goes_ds.CMI_C15.astype(np.float32).compute()

out_dims = goes_ds.CMI_C13.dims
out_coords = goes_ds.CMI_C13.coords

goes_ds.close()

wvd = (C8_data - C10_data)
swd = (C13_data - C15_data)

C8_data.close()
C10_data.close()
C13_data.close()
C15_data.close()

# Calculate flow field using channel 13 BT
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

bt_dt = slots.get_flow_diff(C13_data, flow, dtype=np.float32)/dt[:,np.newaxis,np.newaxis]

filter_thresh = 5
filter_range = 11
delta_bt = C13_data.data - ndi.morphology.grey_erosion(C13_data.data, (1, filter_range, filter_range)).astype(np.float32)
bt_filter = np.maximum((filter_thresh-delta_bt)/filter_thresh,0).astype(np.float32)

bt_growth = (np.maximum(-bt_dt,0) * bt_filter).astype(np.float32)

del dt, bt_dt, delta_bt, bt_filter

upper_thresh = -5
lower_thresh = -15
growth_thresh = 0.5
markers = np.logical_and(wvd.data>upper_thresh,  bt_growth>growth_thresh)
markers = markers.compute().astype('bool')

mask = (wvd.data<lower_thresh)
mask = ndi.grey_erosion(mask, (np.minimum(5, (mask.shape[0]-1)//2), 5, 5)).astype('bool')
mask = np.logical_or(mask, np.isnan(wvd))
inner_edges = slots.flow_sobel(np.minimum(np.maximum(
                                da.nanmax(slots.flow_convolve(wvd - swd
                                    + (bt_growth * 5), flow), 0),
                            lower_thresh), upper_thresh),
                        flow, direction='uphill', magnitude=True,
                        dtype=np.float32).compute()

outer_edges = slots.flow_sobel(np.minimum(np.maximum(
                                da.nanmax(slots.flow_convolve(wvd + swd
                                    + (bt_growth * 5) - 7.5, flow), 0),
                            lower_thresh), upper_thresh),
                        flow, direction='uphill', magnitude=True,
                        dtype=np.float32).compute()

ds = xr.Dataset({'inner_edges':xr.DataArray(inner_edges.astype(np.float32),
                                            dims=out_dims, coords=out_coords)\
                                            .isel({'t':slice(1,-1)}),
                 'outer_edges':xr.DataArray(outer_edges.astype(np.float32),
                                            dims=out_dims, coords=out_coords)\
                                            .isel({'t':slice(1,-1)}),
                 'growth':xr.DataArray(bt_growth.astype(np.float32),
                                       dims=out_dims, coords=out_coords)\
                                       .isel({'t':slice(1,-1)}),
                 'markers':xr.DataArray(markers.astype(bool),
                                        dims=out_dims, coords=out_coords)\
                                        .isel({'t':slice(1,-1)}),
                 'mask':xr.DataArray(mask.astype(bool),
                                     dims=out_dims, coords=out_coords)\
                                     .isel({'t':slice(1,-1)}),
                 'flow_x_for':xr.DataArray(flow.flow_x_for.astype(np.float32),
                                           dims=out_dims, coords=out_coords)\
                                           .isel({'t':slice(1,-1)}),
                 'flow_y_for':xr.DataArray(flow.flow_y_for.astype(np.float32),
                                           dims=out_dims, coords=out_coords)\
                                           .isel({'t':slice(1,-1)}),
                 'flow_x_back':xr.DataArray(flow.flow_x_back.astype(np.float32),
                                            dims=out_dims, coords=out_coords)\
                                            .isel({'t':slice(1,-1)}),
                 'flow_y_back':xr.DataArray(flow.flow_y_back.astype(np.float32),
                                            dims=out_dims, coords=out_coords\
                                            ).isel({'t':slice(1,-1)}),
                 })

save_name = 'slots_preprocess_C_'+date.isoformat()

# save_path =  ('/'.join([save_dir, str(date.year).zfill(4),
#                         str(date.month).zfill(2), str(date.day).zfill(2)]))
#
# if not os.path.isdir(save_path):
#     os.makedirs(save_path)

print(datetime.now(), 'Saving file to:')
print(save_dir + '/' + save_name + '.nc')
ds.to_netcdf(save_dir + '/' + save_name + '.nc', format='NETCDF4')

print(datetime.now(), 'Preprocessing successfully completed')
