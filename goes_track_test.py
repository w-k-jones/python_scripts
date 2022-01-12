#!/home/users/wkjones/miniconda2/envs/NEW/bin/python2.7
from __future__ import print_function, division
import numpy as np
from numpy import ma
import xarray as xr
from glob import glob
from scipy.ndimage.morphology import grey_erosion, grey_dilation, grey_opening, grey_closing
from skimage.morphology import watershed
from scipy.ndimage.filters import sobel
from scipy.ndimage import label
import argparse
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description='Detect deep convective features in GOES-16 ABI imagery')
parser.add_argument('DoY', help='Day of year to process', type=int)
parser.add_argument('-C', help='Use CONUS scan (5 minute) data', action='store_true', default=False)
parser.add_argument('-F', help='Use Full scan (15 minute) data', action='store_true', default=False)
parser.add_argument('-l', help='Contraction length scale: takes the max value over l*l pixels', default=1)
parser.add_argument('-hd', help='Add half-day offset', action='store_true', default=False)


args = parser.parse_args()
doy_str = str(args.DoY).zfill(3)
print('DoY: ', doy_str)
if args.C:
    scan_type = 'C'
if args.F:
    scan_type = 'F'
if not args.C and not args.F:
    raise Exception('No scan type specfied. Please use "-C" or "-F" options to specify scan type')

length_scale = args.l

def get_abi_IR(dataset):
    bt = (dataset.planck_fk2.data / (np.log((dataset.planck_fk1.data / dataset.Rad.data) + 1)) - dataset.planck_bc1.data) / dataset.planck_bc2.data
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

def get_wvd_from_files(C8_file, C10_file):
    with xr.open_dataset(C8_file) as C8_ds:
        C8_BT = get_abi_IR(C8_ds)
    with xr.open_dataset(C10_file) as C10_ds:
        C10_BT = get_abi_IR(C10_ds)
    wvd = C8_BT - C10_BT
    return wvd

def sobel_3d(wvd):
    edges = np.sqrt(sobel(wvd,1)**2 + sobel(wvd,2)**2 + sobel(wvd,0)**2)
    return edges

def mark_threshold(wvd, background=-10, threshold=-1):
    markers = np.full_like(wvd, 0, dtype='uint8')
    markers += 2*(wvd>threshold).astype('uint8')
    markers += grey_erosion((wvd<=background), size=(11,11)).astype('uint8')
    return markers

def get_area_max(x_in, l=1, nanmax=False):
    x = x_in[:(x_in.shape[0]//l)*l,:(x_in.shape[1]//l)*l]
    x = x.reshape(x_in.shape[0]//l, l, x_in.shape[1]//l, l)
    x = np.moveaxis(x, 1, 2).reshape(x_in.shape[0]//l, x_in.shape[1]//l, l**2)
    if nanmax:
        x = np.nanmax(x, axis=2)
    else:
        x = x.max(2)
    return x

def get_goes_abi_files(input_file):
    datestr = input_file.split('/')[-1].split('_')[3]
    yearstr = datestr[1:5]
    doystr = datestr[5:8]
    hourstr = datestr[8:10]
    minstr = datestr[10:12]
    secstr = datestr[12:14]
    file_date = datetime(year=int(yearstr), month=1, day=1, hour=int(hourstr), minute=int(minstr), second=int(secstr)) + timedelta(days=int(doystr)-1)
    files = glob('/'.join(input_file.split('/')[:-1])+'/'+input_file.split('/')[-1].split('_')[0][:-2]+'*s'+yearstr+doystr+hourstr+minstr+'*.nc')
    return [file_date]+files

file_chunk = 288
feature_chunk = 144


goes_path = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES16/Rad'+scan_type+'/2018/'
if args.hd:
    C08_files = glob(goes_path+doy_str+'/1[2-9]/*C08_*.nc') + glob(goes_path+doy_str+'/2[0-3]/*C08_*.nc') + glob(goes_path+str(args.DoY+1).zfill(3)+'/0[0-9]/*C08_*.nc') + glob(goes_path+str(args.DoY+1).zfill(3)+'/1[0-1]/*C08_*.nc')
else:
    C08_files = glob(goes_path+doy_str+'/*/*C08_*.nc')
C08_files.sort()

C_files = []
for C08_file in C08_files:
    files = get_goes_abi_files(C08_file)
    if len(files) == 17:
        C_files.append(files)

with xr.open_dataset(C_files[0][8]) as test_file:
    test_BT = get_abi_IR(test_file)

n_files = len(C_files)
print('files: ', n_files)

test_BT = get_area_max(test_BT, l=length_scale, nanmax=True)

edges = np.full((n_files,)+test_BT.shape, 0, dtype='float16')
markers = np.full((n_files,)+test_BT.shape, 0, dtype='uint8')
temp_wvd = np.full((3,)+test_BT.shape, -10, dtype='float64')
for i in range(n_files):
    wvd = get_wvd_from_files(C_files[i][8], C_files[i][10])
    ird = wvd+ get_wvd_from_files(C_files[i][15], C_files[i][13])
    ird[ird < -15] = -15
    with xr.open_dataset(C_files[i][14]) as C14:
        BT = get_abi_IR(C14)
    markers[i] = 2*np.logical_and(wvd > -5, BT < 250)
    markers[i] += grey_erosion(ird <= -15, size=(11,11))
    markers[i][markers[i] > 2] = 0
    if i < 3:
        temp_wvd[i] = ird
        if i == 2:
            edges[:i] = sobel_3d(grey_dilation(temp_wvd,size=(1,3,3)))[:-1].astype('float16')
    else:
        temp_wvd[:-1] = temp_wvd[1:]
        temp_wvd[-1] = ird
        if i < n_files-1:
            edges[i-1] = sobel_3d(grey_dilation(temp_wvd,size=(1,3,3)))[1].astype('float16')
        else:
            edges[i-1:] = sobel_3d(grey_dilation(temp_wvd,size=(1,3,3)))[1:].astype('float16')

print('watershedding')
features = watershed(edges, markers)
print(np.sum(features > 1.5))
features = features > 1
#markers[grey_erosion(temp_features, size=(1,3,3)) > 1.5] = 2

print('labelling')
labelled_features = label(features>0)[0]

print('Number of regions:', np.nanmax(labelled_features))
print('Total feature pixels:', np.sum(features >= 1))
print('Total background pixels:', np.sum(features == 0))

print('saving')
save_dir = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES16/features/'
savename = 'Rad'+scan_type+'_features_'+doy_str
if args.hd:
    savename+='hd'
#np.save('./test_features', features)
np.savez(save_dir+savename, files=C_files, markers=markers.astype('uint8'), features=features.astype('uint8'), labels=labelled_features.astype('uint16'))
