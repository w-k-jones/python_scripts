#!/home/users/wkjones/miniconda2/envs/slot_process/bin/python
from __future__ import print_function, division
import numpy as np
from numpy import ma
import xarray as xr
from glob import glob
from datetime import datetime, timedelta
from scipy import ndimage as ndi
from dateutil.parser import parse as parse_date
import argparse
from python_toolbox import abi_tools, dataset_tools, opt_flow
import cv2 as cv



parser = argparse.ArgumentParser(description="""Detect deep convective features
    in GOES-16 ABI imagery using a semi-lagrangian method""")
parser.add_argument('satellite', help='GOES satellite to use (16 or 17)', type=int)
parser.add_argument('start_date', help='Start date of processing', type=str)
parser.add_argument('end_date', help='End date of processing', type=str)
parser.add_argument('-C', help='Use CONUS scan (5 minute) data', action='store_true', default=False)
parser.add_argument('-F', help='Use Full scan (15 minute) data', action='store_true', default=False)
parser.add_argument('-x0', help='Initial subset x location', default=0, type=int)
parser.add_argument('-x1', help='End subset x location', default=-1, type=int)
parser.add_argument('-y0', help='Initial subset y location', default=0, type=int)
parser.add_argument('-y1', help='End subset y location', default=-1, type=int)

#parser.add_argument('-l', help='Contraction length scale: takes the max value over l*l pixels', default=1)

args = parser.parse_args()
satellite = str(args.satellite)
start_date = parse_date(args.start_date)
end_date = parse_date(args.end_date)
x0 = int(args.x0)
x1 = int(args.x1)
y0 = int(args.y0)
y1 = int(args.y1)

print(datetime.now(),'Detecting features for satellite GOES'+satellite)
print('Start date:', start_date.isoformat())
print('End date:', end_date.isoformat())

base_path = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES'+satellite+'/Rad'
if args.C:
    scan_type = 'C'
    base_path = base_path + 'C/'
elif args.F:
    scan_type = 'F'
    base_path = base_path + 'F/'
else:
    raise ValueError("""Error in abi_wvd_detect: -C or -F input must be selected""")

print(datetime.now(),'Finding files in:', base_path)

file_names = []
doy_start = (start_date - datetime(start_date.year, 1, 1)).days + 1
doy_end = (end_date - datetime(end_date.year, 1, 1)).days + 1

# Load files -- glob the channel 8 files first
for year in range(start_date.year, end_date.year+1):
    year_str = str(year).zfill(4)
    if start_date.year == end_date.year:
        for doy in range(doy_start, doy_end+1):
            doy_str = str(doy).zfill(3)
            goes_path = base_path +year_str+'/'+doy_str+'/'
            file_names.extend(glob(goes_path+'*/*C08_*.nc'))
    elif year == start_date.year:
        for doy in range(doy_start, (datetime(year+1,1,1)-datetime(year,1,1)).days+1):
            doy_str = str(doy).zfill(3)
            goes_path = base_path +year_str+'/'+doy_str+'/'
            file_names.extend(glob(goes_path+'*/*C08_*.nc'))
    elif year == end_date.year:
        for doy in range(1, doy_end+1):
            doy_str = str(doy).zfill(3)
            goes_path = base_path +year_str+'/'+doy_str+'/'
            file_names.extend(glob(goes_path+'*/*C08_*.nc'))
    else:
        for doy in range(1, (datetime(year+1,1,1)-datetime(year,1,1)).days+1):
            doy_str = str(doy).zfill(3)
            goes_path = base_path +year_str+'/'+doy_str+'/'
            file_names.extend(glob(goes_path+'*/*C08_*.nc'))

# Get all filenames
goes_files = [abi_tools.get_goes_abi_files(f) for f in file_names]
# Filter for start and end date
goes_files = [f for f in goes_files if f[0] >= start_date and f[0] <= end_date and len(f) == 17]

print('Files discovered:', len(goes_files))
if len(goes_files) == 0:
    raise ValueError("""Error in abi_wvd_detect: No abi files discovered!""")
# Now load the data
file_slice = {'y':slice(y0, y1), 'x':slice(x0, x1)}
print(datetime.now(),'Loading data from channel 8')
C8_data = abi_tools.get_abi_ds_from_files([f[8] for f in goes_files], dtype='float32', slices=file_slice)
print(datetime.now(),'Loading data from channel 10')
C10_data = abi_tools.get_abi_ds_from_files([f[10] for f in goes_files], dtype='float32', slices=file_slice)
print(datetime.now(),'Loading data from channel 13')
C13_data = abi_tools.get_abi_ds_from_files([f[13] for f in goes_files], dtype='float32', slices=file_slice)
print(datetime.now(),'Loading data from channel 15')
C15_data = abi_tools.get_abi_ds_from_files([f[15] for f in goes_files], dtype='float32', slices=file_slice)

# Match the time coordinates to make arithmetic work nicely
dataset_tools.match_coords([C8_data,C10_data,C13_data,C15_data])
out_dims = C13_data.dims
out_coords = C13_data.coords

# Calculate the water vapour and split window differences
print(datetime.now(), 'Computing water vapour and split window differences')
wvd = (C8_data - C10_data)
swd = (C13_data - C15_data)

C8_data.close()
C10_data.close()
C15_data.close()

# Calculate flow field using channel 13 BT
print(datetime.now(), 'Predicting flow')
flow_kwargs = {'pyr_scale':0.5, 'levels':7, 'winsize':64, 'iterations':4,
               'poly_n':7, 'poly_sigma':1.5, 'flags':cv.OPTFLOW_FARNEBACK_GAUSSIAN}
# field_flow = opt_flow.get_flow_func(C13_data, replace_missing=True, dtype='float32')
C13_res, C13_rel, C13_flow = opt_flow.test_flow(C13_data, return_flow=True, replace_missing=True, post_iter=3, **flow_kwargs)
wvd_res, wvd_rel, wvd_flow = opt_flow.test_flow(wvd, return_flow=True, replace_missing=True, post_iter=3, **flow_kwargs)
swd_res, swd_rel, swd_flow = opt_flow.test_flow(swd, return_flow=True, replace_missing=True, post_iter=3, **flow_kwargs)

weights = np.stack([np.maximum(1-C13_rel, 0.01)*C13_res/C13_rel,
                    np.maximum(1-wvd_rel, 0.01)*wvd_res/wvd_rel,
                    np.maximum(1-swd_rel, 0.01)*swd_res/swd_rel], 0)

field_flow = opt_flow.Flow_Func(np.average(np.stack([C13_flow.flow_x_for,
                                              wvd_flow.flow_x_for,
                                              swd_flow.flow_x_for], 0),
                                     axis=0, weights=weights),
                          np.average(np.stack([C13_flow.flow_x_back,
                                              wvd_flow.flow_x_back,
                                              swd_flow.flow_x_back], 0),
                                     axis=0, weights=weights),
                          np.average(np.stack([C13_flow.flow_y_for,
                                              wvd_flow.flow_y_for,
                                              swd_flow.flow_y_for], 0),
                                     axis=0, weights=weights),
                          np.average(np.stack([C13_flow.flow_y_back,
                                              wvd_flow.flow_y_back,
                                              swd_flow.flow_y_back], 0),
                                     axis=0, weights=weights))

del C13_res, C13_rel, C13_flow, wvd_res, wvd_rel, wvd_flow, swd_res, swd_rel, swd_flow

print(datetime.now(), 'Calculating Growth Metric')
dx, dy = abi_tools.get_abi_pixel_lengths(C13_data) # in km
dt = [(goes_files[1][0]-goes_files[0][0]).total_seconds()/60] \
     + [(goes_files[i+2][0]-goes_files[i][0]).total_seconds()/120 for i in range(len(goes_files)-2)] \
     + [(goes_files[-1][0]-goes_files[-2][0]).total_seconds()/60]
dt = np.array(dt)

bt_stack = opt_flow.get_flow_stack(C13_data, field_flow).data.rechunk((-1,C13_data.shape[0],-1,-1))
bt_dt = np.gradient(bt_stack, axis=0)[1]/dt[:,np.newaxis,np.newaxis]# bt_grad = np.gradient(np.gradient(bt_stack[1], axis=1), axis=1)/dy**2 + np.gradient(np.gradient(bt_stack[1], axis=2), axis=2)/dx**2

filter_thresh = 5
filter_range = 11
delta_bt = bt_stack[1] - ndi.morphology.grey_erosion(bt_stack[1], (1, filter_range, filter_range))
bt_filter = np.maximum((filter_thresh-delta_bt)/filter_thresh,0)
bt_growth = np.maximum(-bt_dt,0)*bt_filter

# Thresholds, should put these as optional inputs
upper_thresh = -5
lower_thresh = -15
growth_thresh = 0.5
print(datetime.now(), 'Calculating markers')
markers = wvd>upper_thresh
markers = np.logical_and(markers, bt_growth>growth_thresh).to_masked_array()
print(datetime.now(), 'Calculating mask')
mask = (wvd<lower_thresh).to_masked_array()
for i in range(np.minimum(3, (mask.shape[0]-1)//2)):
    # Semi-lagrangian erosion of the mask by up to three steps
    mask = opt_flow.flow_convolve_nearest(mask, field_flow,
                                          structure=ndi.generate_binary_structure(3,2),
                                          function=np.nanmin)

print(datetime.now(), 'Freeing memory')
C13_data.close()
del bt_stack, delta_bt, bt_filter, bt_dt
import gc
gc.collect()

# Calculate edges
print(datetime.now(), 'Calculating inner edges')
inner_stack = np.minimum(np.maximum(
                            opt_flow.flow_convolve_nearest((wvd-swd+(bt_growth*5)-5).data,
                                                           field_flow,
                                                           function=np.nanmax),
                                    lower_thresh), upper_thresh)
# Get edges
inner_edges = opt_flow.flow_sobel(opt_flow.get_flow_stack(
                                      xr.DataArray(inner_stack,
                                          dims=('t','y','x')),
                                      field_flow).to_masked_array(),
                                  direction='uphill', magnitude=True)

del inner_stack
gc.collect()

inner = opt_flow.flow_network_watershed(inner_edges, markers.filled(), field_flow, mask=mask.filled(), debug_mode=True, max_iter=25)

del inner_edges
gc.collect()

print(datetime.now(), 'Calculating outer edges')
outer_stack = np.minimum(np.maximum(
                            opt_flow.flow_convolve_nearest((wvd+swd+(bt_growth*5)-7.5).data,
                                                           field_flow,
                                                           function=np.nanmax),
                                    lower_thresh), upper_thresh)
# Get edges
outer_edges = opt_flow.flow_sobel(opt_flow.get_flow_stack(
                                      xr.DataArray(outer_stack,
                                          dims=('t','y','x')),
                                      field_flow).to_masked_array(),
                                  direction='uphill', magnitude=True)

del outer_stack
gc.collect()

print(datetime.now(), 'Finding outer features')
outer = opt_flow.flow_network_watershed(outer_edges, markers.filled(), field_flow, mask=mask.filled(), debug_mode=True, max_iter=25)

# print(datetime.now(), 'Masking regions')
# regions = ma.array(np.maximum(regions,0), mask=outer<=0)

del outer_edges
gc.collect()

# print('Total background pixels:', np.sum(regions == 0))

print(datetime.now(), 'saving')
# save_dir = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES16/features/'
# savename = 'abi_Rad'+scan_type+'_flow_features_S'+start_date.isoformat()+'_E'+end_date.isoformat()
# print(save_dir+savename+'.npz')
# # np.savez(save_dir+savename, files=goes_files, labels=features.astype(bool), outer=outer.astype(bool), growth=bt_growth.astype(np.float32))
# np.savez(save_dir+savename, files=goes_files, labels=outer.astype(bool), growth=bt_growth.astype(np.float32))

ds = xr.Dataset({'inner':xr.DataArray(inner.astype(bool), dims=out_dims, coords=out_coords),
                 'outer':xr.DataArray(outer.astype(bool), dims=out_dims, coords=out_coords),
                 'growth':xr.DataArray(bt_growth, dims=out_dims, coords=out_coords)})

save_name = 'abi_Rad'+scan_type+'_flow_features_S'+start_date.isoformat()+'_E'+end_date.isoformat()
save_dir = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES'+satellite+'/features/'

ds.to_netcdf(save_dir+save_name+'.nc', format='NETCDF4')

print(datetime.now(), 'Feature detection successfully completed')
