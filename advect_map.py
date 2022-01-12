import trajectory_tools as tt
import seviri_tools as st
import numpy as np
from glob import glob
from scipy import interpolate
import datetime
import netCDF4 as nc
import matplotlib.pyplot as plt
import html_animation

sev_files = glob('/group_workspaces/jasmin2/acpc/Data/ORAC/clarify/merged/*SEVIRI*2017090[23]*.merged.nc')
sev_files.sort()
sev_dates = st.get_seviri_file_time(sev_files)
# Offset by 5 minutes for rough location
sev_dates = [date + datetime.timedelta(minutes=5) for date in sev_dates]

traj_files = glob('/home/users/wkjones/data/hysplit_traj/tdump.*')
trajectories = tt.merge_traj([tt.parse_hysplit(f) for f in traj_files])
t_shape = trajectories.shape
interp_locs = [t.get_loc_at_t(sev_dates, extrapolate=True) for t in trajectories.flatten()]
lons = np.stack([i[0] for i in interp_locs], axis=-1).reshape((-1,) + t_shape)
lats = np.stack([i[1] for i in interp_locs], axis=-1).reshape((-1,) + t_shape)

res = 100
# index locations for trajectories
ilon = range(t_shape[1])
ilat = range(t_shape[0])
# grid locations to interpolate high res locations to
glon, glat = np.meshgrid(np.linspace(0,t_shape[1]-1,res*(t_shape[1]-1)+1),
                         np.linspace(0,t_shape[0]-1,res*(t_shape[0]-1)+1))

variables = ['reflectance_in_channel_no_1','brightness_temperature_in_channel_no_10','ctp','ctt','cer','cot']
vrange = [(0.,1.), (275.,300.), (800.,1000.), (250.,300.), (0.,40.), (0.,40.)]
cmap = ['gray', 'inferno', 'viridis', 'inferno', 'inferno_r', 'viridis']
unit = ['','/K','/hPa','/K',r'/$\mu$m','']
# Reduce to 1 for first test

variables = ['brightness_temperature_in_channel_no_4']
vrange = [(275.,300.)]
cmap = ['inferno']
unit = ['/K']

out_files = [[] for i in range(len(variables))]

out_path = '/group_workspaces/jasmin2/cloud_ecv/public/temp_wkjones/advect'

for i, f in enumerate(sev_files):
    interp_lon = interpolate.RectBivariateSpline(ilat, ilon, lons[i])
    hr_lon = interp_lon.ev(glat, glon)
    interp_lat = interpolate.RectBivariateSpline(ilat, ilon, lats[i])
    hr_lat = interp_lat.ev(glat, glon)
    sev_x, sev_y = st.map_ll_to_seviri(hr_lon, hr_lat)
    latrange = [np.floor(np.min(hr_lat)/5)*5,
                np.ceil(np.max(hr_lat)/5)*5]
    lonrange = [np.floor(np.min(hr_lon)/5)*5,
                np.ceil(np.max(hr_lon)/5)*5]
    nc_sev = nc.Dataset(f)
    if f == sev_files[0]:
        lat_orac = nc_sev.variables['lat'][:]
        lon_orac = nc_sev.variables['lon'][:]
    wh = np.where(
            np.logical_and(
            np.logical_and(lat_orac>=latrange[0],lat_orac<=latrange[1]),
            np.logical_and(lon_orac>=lonrange[0],lon_orac<=lonrange[1])))
    irange = [np.min(wh[0]), np.max(wh[0])+1]
    jrange = [np.min(wh[1]), np.max(wh[1])+1]
    for j, varname in enumerate(variables):
        var = nc_sev.variables[varname][irange[0]:irange[1], jrange[0]:jrange[1]]
        var_interp = interpolate.RectBivariateSpline(np.arange(irange[0], irange[1]), np.arange(jrange[0], jrange[1]), var)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(var_interp.ev(sev_y, sev_x), vmin=vrange[j][0], vmax=vrange[j][1], cmap=cmap[j], origin='bottom')
        cb = plt.colorbar(im)
        cb.set_label(unit[j])
        ax.set_title('SEVIRI '+varname+' '+f.split('/')[-1][38:50])
        ax.set_xlabel('Easterly')
        ax.set_ylabel('Northerly')
        savename = out_path+'/'+varname+'_'+f.split('/')[-1]+'.png'
        print 'Plotting to: '+savename
        plt.savefig(savename)
        out_files[j].append(savename)
        plt.close()


for i, varname in enumerate(variables):
    out_files[i].sort()
    html_animation.html_script([p.split('/')[-1] for p in out_files[i]],
                                        varname, out_path+'/', make_gif='yes')
