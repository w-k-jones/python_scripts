#!/home/users/wkjones/miniconda2/envs/slot_process/bin/python
import numpy as np
from numpy import ma
from python_toolbox import abi_tools
import xarray as xr
from scipy import stats
from glob import glob
import pyart
from pyproj import Proj
from dateutil.parser import parse as parse_date
from datetime import datetime, timedelta
import tarfile
import argparse

parser = argparse.ArgumentParser(description="""Regrid nexrad radar reflectivity
    to the GOES-ABI imager projection""")
parser.add_argument('date', help='Start date of processing', type=str)
args = parser.parse_args()
date = parse_date(args.date)

yr_str = str(date.year)
mo_str = str(date.month).zfill(2)
dy_str = str(date.day).zfill(2)
hr_str = str(date.hour).zfill(2)
mn_str = str(date.minute).zfill(2)
sc_str = str(date.second).zfill(2)

date_str = yr_str+mo_str+dy_str+hr_str+mn_str+sc_str

doy_str = str((date-datetime(date.year,1,1)).days+1).zfill(3)

goes_path = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES16/RadC/'+yr_str+'/'+doy_str+'/'+hr_str+'/'
goes_files = sorted(glob(goes_path+'OR_ABI-L1b-RadC-M[36]C13_G16_*.nc'))
# print(goes_path)
# print(goes_files)
goes_files = [abi_tools.get_goes_abi_files(f) for f in goes_files]

print(goes_files)
if len(goes_files) == 0:
    raise ValueError("No ABI files discovered")

C13_data = abi_tools.get_abi_ds_from_files(goes_files[0][13])
shape = (len(goes_files),) + C13_data.shape[-2:]
p = abi_tools.get_abi_proj(C13_data)
height = C13_data.goes_imager_projection.perspective_point_height
lat0 = C13_data.goes_imager_projection.latitude_of_projection_origin
lon0 = C13_data.goes_imager_projection.longitude_of_projection_origin
goes_dates = [f[0] for f in goes_files]
# print(goes_dates)

x_bins = np.zeros(C13_data.x.size+1)
x_bins[:-1] += C13_data.x
x_bins[1:] += C13_data.x
x_bins[1:-1] /= 2
x_bins *= height

y_bins = np.zeros(C13_data.y.size+1)
y_bins[:-1] += C13_data.y
y_bins[1:] += C13_data.y
y_bins[1:-1] /= 2
y_bins *= height

radar_dir = '/gws/nopw/j04/eo_shared_data_vol2/scratch/radar/nexrad_l2/'+yr_str+'/'+mo_str+'/'+dy_str+'/'
radar_dirs = glob(radar_dir+'*')
radar_dirs = [rd for rd in radar_dirs if rd.split('/')[-1][0] == 'K' or rd.split('/')[-1][0] == 'T']

radar_tarfiles = sum([glob(rd+'/*_'+date_str+'_*.tar') for rd in radar_dirs],[])

grid_sum = np.zeros(shape)
grid_max = np.zeros(shape)
grid_count = np.zeros(shape)

for tf in radar_tarfiles:
    print(tf)
    with tarfile.open(tf) as tar:
        x_list, y_list, ref_list, t_list = [],[],[],[]
        for item in [name for name in tar.getnames() if name[-9:] == '_V06.ar2v']:
            try:
                radar = pyart.io.read_nexrad_archive(tar.extractfile(tar.getmember(item)))
            except IOError:
                pass
            else:
                start_time = parse_date(item[4:19], fuzzy=True)

                rad_time = [start_time+timedelta(seconds=t) for t in radar.time['data']]
                rad_alt = radar.gate_altitude['data']
                rad_lat = radar.gate_latitude['data']
                rad_lon = radar.gate_longitude['data']
                rad_data = radar.fields['reflectivity']['data']
                rad_x, rad_y = p(rad_lon, rad_lat)
#         New parallax "uncorrection"
                dlat = np.degrees(rad_alt*np.tan(np.radians(rad_lat-lat0) + rad_y/height)/6.371E6)
                dlon = np.degrees(rad_alt*np.tan(np.radians(rad_lon-lon0) + rad_x/height)/6.371E6)
                rad_x, rad_y = p(rad_lon+dlon, rad_lat+dlat)

                mask = np.any([rad_data.mask,
                               rad_x>(C13_data.x.max().data*height),
                               rad_x<(C13_data.x.min().data*height),
                               rad_y>(C13_data.y.max().data*height),
                               rad_y<(C13_data.y.min().data*height),
                               rad_alt<500, rad_alt>15000], 0)

                t = ma.array(np.tile(rad_time, (1,rad_lat.shape[1])), mask=mask)
                y = ma.array(rad_y, mask=mask)
                x = ma.array(rad_x, mask=mask)
                ref = ma.array(rad_data, mask=mask)
                if not np.all(mask):
                    x_list.append(x.compressed())
                    y_list.append(y.compressed())
                    t_list.append(t.compressed())
                    ref_list.append(ref.compressed())
    if len(x_list)>0:
        x = ma.concatenate(x_list, 0)
        y = ma.concatenate(y_list, 0)
        t = ma.concatenate(t_list, 0)
        ref = ma.concatenate(ref_list, 0)

        if x.size>0:
            for i, time in enumerate(goes_dates):
                wh_t = np.logical_and(t>time, t<(time+timedelta(minutes=5)))
                if np.any(wh_t):
                    grid_sum[i] += ma.array(stats.binned_statistic_dd((y[wh_t], x[wh_t]),
                                                                      ref[wh_t],
                                                                      statistic='sum',
                                                                      bins=(y_bins[::-1], x_bins),
                                                                      expand_binnumbers=True)[0][::-1])
                    # This is too slow at the current time. Maybe need to make a custom script
                    # grid_max[i] = np.fmax(grid_max[i],
                    #                       ma.array(stats.binned_statistic_dd((y[wh_t], x[wh_t]),
                    #                                                   ref[wh_t],
                    #                                                   statistic='max',
                    #                                                   bins=(y_bins[::-1], x_bins),
                    #                                                   expand_binnumbers=True)[0][::-1]))
                    grid_count[i] += ma.array(np.histogramdd([y[wh_t], x[wh_t]],
                                                             bins=(y_bins[::-1], x_bins))[0][::-1])
    if len(x_list)>0:
        del x,y,t,ref
    del x_list, y_list, ref_list, t_list

ds = xr.Dataset({'reflectivity_mean':xr.DataArray(grid_sum/grid_count, dims=('t','y','x'),
                                             coords={'t':goes_dates, 'y':C13_data.y, 'x':C13_data.x}),
                 # 'reflectivity_max':xr.DataArray(grid_max, dims=('t','y','x'),
                 #                             coords={'t':goes_dates, 'y':C13_data.y, 'x':C13_data.x}),
                 'counts':xr.DataArray(grid_count, dims=('t','y','x'),
                                       coords={'t':goes_dates, 'y':C13_data.y, 'x':C13_data.x})})

save_name = 'nexrad_goes_regrid_'+date_str
save_dir = '/gws/nopw/j04/eo_shared_data_vol2/scratch/radar/nexrad_regrid/'

ds.to_netcdf(save_dir+save_name+'.nc', format='NETCDF4')
print(save_dir+save_name+'.nc')
