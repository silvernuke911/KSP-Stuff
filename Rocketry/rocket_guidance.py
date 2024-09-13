import numpy as np
import matplotlib.pyplot as plt
from ksp_planets import *
import vector_operations as vc

theta = np.linspace(0,2*np.pi,1001)
radius = np.full((1001),kerbin.equatorial_radius)
mun_shape = np.array([[mun.equatorial_radius*np.cos(angle),mun.equatorial_radius*np.sin(angle)] for angle in theta])
kerbin_shape = np.array([kerbin.equatorial_radius*np.array([np.cos(angle),np.sin(angle)]) for angle in theta])
kerbin_atmo_shape = np.array([(kerbin.equatorial_radius+7e4)*np.array([np.cos(angle),np.sin(angle)]) for angle in theta])

class Spacecraft:
    def __init___(self):
        self.drymass = 0
        self.wetmass = 0
        self.isp = 0
        self.drag_coefficient = 0
        self.thrust = 0

plt.figure(figsize=(10, 10))
plt.plot(kerbin_shape[:,0],kerbin_shape[:,1],color = 'royalblue')
plt.plot(kerbin_atmo_shape[:,0],kerbin_atmo_shape[:,1],color = 'cyan')
plt.axis('equal')
plt.grid()
plt.show()

plt.plot(np.rad2deg(theta), radius)
plt.xlim([min(np.rad2deg(theta)), max(np.rad2deg(theta))])
plt.grid()
plt.show()

