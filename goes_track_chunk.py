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

file_chunk = 288
feature_chunk = 144
length_scale = 1

goes_path = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES16/RadC/2018/170/'
C08_files = glob(goes_path+'0[0-9]/*C08_*.nc')
C10_files = glob(goes_path+'0[0-9]/*C10_*.nc')
C14_files = glob(goes_path+'0[0-9]/*C14_*.nc')
if len(C08_files) != len(C10_files):
    raise Exception('Missing files')

with xr.open_dataset(C08_files[0]) as test_file:
    test_BT = get_abi_IR(test_file)

n_files = len(C08_files)
print('Number of files: ', n_files)
if n_files <= file_chunk:
    file_chunk = n_files
    n_file_chunks = 1
else:
    n_file_chunks = n_files // file_chunk
    if (n_files % file_chunk) != 0:
        n_file_chunks+=1
    n_file_chunks = n_file_chunks*2 - 1
print('File chunk length: ', file_chunk)
print('Number of chunks:', n_file_chunks)

if n_files <= feature_chunk:
    feature_chunk = n_files
    n_feature_chunks = 1
else:
    n_feature_chunks = n_files // feature_chunk
    if (n_files % file_chunk) != 0:
        n_feature_chunks+=1
    n_feature_chunks = n_feature_chunks*2 - 1
print('Feature chunk length: ', feature_chunk)
print('Number of chunks:', n_feature_chunks)



test_BT = get_area_max(test_BT, l=length_scale, nanmax=True)

edges = np.full((n_files,)+test_BT.shape, 0, dtype='float16')
markers = np.full((n_files,)+test_BT.shape, 0, dtype='uint8')
features = np.full((n_files,)+test_BT.shape, 0, dtype='uint8')
temp_wvd = np.full((3,)+test_BT.shape, -10, dtype='float64')
for i in range(n_files):
    wvd = get_area_max(get_wvd_from_files(C08_files[i], C10_files[i]), l=length_scale, nanmax=True)
    wvd[wvd <=-10] = -10
    if i < 3:
        temp_wvd[i] = wvd
        markers[i] = mark_threshold(temp_wvd[i], threshold=-2.5)
        if i == 2:
            edges[:i] = sobel_3d(grey_dilation(temp_wvd,size=(1,3,3)))[:-1].astype('float16')
    else:
        temp_wvd[:-1] = temp_wvd[1:]
        temp_wvd[-1] = wvd
        markers[i] = mark_threshold(temp_wvd[-1], threshold=-2.5)
        if i < n_files-1:
            edges[i-1] = sobel_3d(grey_dilation(temp_wvd,size=(1,3,3)))[1].astype('float16')
        else:
            edges[i-1:] = sobel_3d(grey_dilation(temp_wvd,size=(1,3,3)))[1:].astype('float16')

print('watershedding')
for i in range(n_chunks):
    print('Chunk: ', i)
    if i < (n_chunks-1):
        start = (i*chunk_length)//2
        end = start + chunk_length
        temp_features = watershed(edges[start:end], markers[start:end])
        print(np.sum(temp_features > 1.5))
        features[start:end][temp_features > 1.5] = 1
        markers[start:end][grey_erosion(temp_features, size=(1,3,3)) > 1.5] = 2
    else:
        start = (i*chunk_length)//2
        temp_features = watershed(edges[start:], markers[start:])
        print(np.sum(temp_features > 1.5))
        features[start:][temp_features > 1.5] = 1

print('labelling')
labelled_features = label(features>0)[0]

print('Number of regions:', np.nanmax(labelled_features))
print('Total feature pixels:', np.sum(features >= 1))
print('Total background pixels:', np.sum(features == 0))

print('saving')
#np.save('./test_features', features)
np.savez('./test_features_no_chunk', features, labelled_features)
