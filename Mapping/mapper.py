import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.font_manager as font_manager
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'axes.formatter.use_mathtext': True,
    'font.size': 10
})

mpl.rcParams['savefig.dpi'] = 700

cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')

# Read the CSV file
filename = 'ksc_area3.csv'
data = pd.read_csv(filename)
step_size = 1 / 50 # define for new values

# Assuming the CSV file has columns named 'latitude', 'longitude', and 'height'
latitudes = data['latitude'].values
longitudes = data['longitude'].values
heights = data['height'].values

# Create a grid for the contour plot

lat_grid, lon_grid = np.meshgrid(
    np.linspace(latitudes.min(), latitudes.max(), round(np.ptp(longitudes) / step_size)),
    np.linspace(longitudes.min(), longitudes.max(), round(np.ptp(latitudes) / step_size)),
)

# Interpolate the heights for the grid
heights_grid = griddata((latitudes, longitudes), heights, (lat_grid, lon_grid), method='cubic')

heightmap = { 
    -1500: 'black',
    -600: 'midnightblue',
    -400: 'navy',       # Deep water
    -200: 'blue',      
    -75   : 'lightsteelblue', # Shallow water
    50  : 'khaki',     #land border
    150 : 'green',       # Land
    400 : 'darkgreen',
    1000: 'darkgoldenrod',    # High land
    1500: 'chocolate',
    2000: 'brown',      # Mountains
    2500: "grey",
    3500: 'dimgray',
    5500: 'tan'
}

minheight = -1500
maxheight = 5500
gap = 1

colors = [((height - minheight) / (maxheight - minheight), color) for height, color in heightmap.items()]
levels = np.arange(minheight, maxheight + gap, gap)
from matplotlib.colors import LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list('height_cmap', colors)



# Plotting the contour plot
fig = plt.figure()
contour = plt.contourf(lon_grid, lat_grid, heights_grid, levels=levels, cmap=custom_cmap, zorder = 0)
plt.colorbar(contour, ticks=np.arange(minheight,maxheight+500,500))

plt.xlabel(r'Longitude ($^{\circ}$)')
plt.ylabel(r'Latitude ($^{\circ}$)')
plt.title('Height Contour Plot : Kerbalia Province')
ax = plt.gca()
ax.set_axisbelow(True)
ax.set_aspect('equal', adjustable='box')

plt.minorticks_on()

# plt.grid(zorder=5, color = 'black')
plt.savefig('ksc_area4.png')
plt.show()

# ax =  plt.axes(projection='3d')
# ax.plot_surface(lon_grid, lat_grid ,heights_grid,cmap=custom_cmap)
# plt.show()

