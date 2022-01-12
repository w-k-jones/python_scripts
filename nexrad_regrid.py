#!/home/users/wkjones/miniconda2/envs/py3/bin/python3.7
import numpy as np
from numpy import ma
import xarray as xr
from scipy import interpolate
from glob import glob
from pyproj import Proj, Geod
import pyart

goes_path = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES16/RadC/2018/170/18/'
goes_files = sorted(glob(goes_path+'OR_ABI-L1b-RadC-M3C13_G16_s201817018*.nc'))

grid_dataset = xr.open_dataset(goes_files[5])
height = grid_dataset.goes_imager_projection.perspective_point_height
g = Geod(ellps='WGS84')
p = Proj(proj='geos', h=grid_dataset.goes_imager_projection.perspective_point_height,
         lon_0=grid_dataset.goes_imager_projection.longitude_of_projection_origin,
         sweep=grid_dataset.goes_imager_projection.sweep_angle_axis)

nexrad_path = '/gws/nopw/j04/eo_shared_data_vol2/scratch/radar/nexrad_l2/2018/06/19/*/'
nexrad_files = sorted(glob(nexrad_path+'*20180619_183[0123]*_V06.ar2v'))
print(nexrad_files)

output_grid = np.full((15,1500,2500), 0.)
count = np.full((15,1500,2500), 0)

for filename in nexrad_files:
    print(filename)
    radar = pyart.io.read_nexrad_archive(filename)
    rad_lons, rad_lats, baz = g.fwd(np.full_like(radar.fields['reflectivity']['data'].data, radar.longitude['data']),
                                    np.full_like(radar.fields['reflectivity']['data'].data, radar.latitude['data']),
                                    *np.meshgrid(radar.azimuth['data'], radar.range['data'], indexing='ij'))
    x, y = p(rad_lons, rad_lats)
    Re = 6.371e6
    h = (radar.range['data']*np.sin(np.radians(radar.elevation['data'][:,np.newaxis]))
             + (Re/np.cos(radar.range['data']/Re)-Re)
             + radar.altitude['data'])
    mask = np.any([x>grid_dataset.x.max().data*height,
                   x<grid_dataset.x.min().data*height,
                   y>grid_dataset.y.max().data*height,
                   y<grid_dataset.y.min().data*height,
                   h<0, h>15000], 0)
    y = ma.array(y, mask=mask)
    x = ma.array(x, mask=mask)
    h = ma.array(h, mask=mask)
    ref = radar.fields['reflectivity']['data'].copy()
    if np.any(np.logical_not(mask)):
        regrid = interpolate.griddata((h.compressed(), y.compressed(), x.compressed()),
                                   ref[np.logical_not(mask)],
                                   np.stack(np.meshgrid(np.arange(500,15000,1000),
                                            grid_dataset.y.data*height,
                                            grid_dataset.x.data*height,
                                            indexing='ij'),-1),
                                   method='nearest')
        wh = np.isfinite(regrid)
        output_grid[wh] += regrid[wh]
        count[wh] += 1

count_g0 = count > 0
output_grid[count_g0] /= count[count_g0]
output_grid = ma.array(output_grid, mask=count==0)
np.save('/home/users/wkjones/regrid_test', output_grid)
