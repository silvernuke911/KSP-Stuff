import numpy as np
import matplotlib.pyplot as plt
from ksp_planets import *
import vector_operations as vc

def drag_acc(m, v, area, C_d, rho):
    return -(0.5/m)*area*C_d*rho* (vc.mag(v))**2 * vc.normalize(v)
def rho(h):
    if h > 68000:
        return 0
    return 1.225*np.exp( -0.00012305 * h)
def a_g(r):
    a = (kerbin.standard_gravitational_parameter) / ((vc.mag(r))**2) * vc.normalize(r)
    return -a
theta = np.linspace(0,2*np.pi,1001)

mun_shape = np.array([[mun.equatorial_radius*np.cos(angle),mun.equatorial_radius*np.sin(angle)] for angle in theta])
kerbin_shape = np.array([kerbin.equatorial_radius*np.array([np.cos(angle),np.sin(angle)]) for angle in theta])
kerbin_atmo_shape = np.array([(kerbin.equatorial_radius+7e4)*np.array([np.cos(angle),np.sin(angle)]) for angle in theta])

class Projectile:
    def __init__(self):
        self.mass = 2000
        self.drag_coefficient = 0.6
        self.area = 2

spacecraft = Projectile()

v_i = np.array([0,2210])
p_i = np.array([-(kerbin.equatorial_radius+10e4),0])

dt = 0.1
max_time = 4e3
t = np.arange(0,max_time+dt,dt)

p = np.zeros((len(t), 2))
v = np.zeros((len(t), 2))
a = np.zeros((len(t), 2))

p[0] = p_i
v[0] = v_i

def height(r):
    return vc.mag(r)-kerbin.equatorial_radius

valid_counter = 0
for i in range(1,len(t)):
    a[i] = a_g(p[i-1]) + drag_acc(spacecraft.mass,v[i-1],spacecraft.area,spacecraft.drag_coefficient,rho(height(p[i-1])))
    v[i] = v[i-1] + a[i] * dt
    p[i] = p[i-1] + v[i] * dt

    valid_counter += 1

    if height(p[i]) < 0 :
        break

p = p[:valid_counter]
v = v[:valid_counter]
a = a[:valid_counter]
t = t[:valid_counter]

a_mag = np.array([vc.mag(i) for i in a])

plt.plot(kerbin_shape[:,0],kerbin_shape[:,1],color = 'royalblue')
plt.plot(kerbin_atmo_shape[:,0],kerbin_atmo_shape[:,1],color = 'cyan')

plt.plot(p[:,0],p[:,1],color = 'red')
plt.plot(v[:,0],v[:,1],color = 'blue')
plt.plot(a[:,0],a[:,1],color = 'green')
plt.plot(t , a_mag ,color = 'brown')
plt.axis('equal')
plt.grid()
plt.show()
