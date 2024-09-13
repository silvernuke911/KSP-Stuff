import numpy as np
import matplotlib.pyplot as plt
import vector_operations as vc
import time
from progress import progress_bar
from ksp_planets import *

def visviva(body,r,a):
    return np.sqrt(body.standard_gravitational_parameter * (2 / r - 1 / a))

def apside_velocity(body, periapsis, apoapsis):
    r1 = periapsis + body.equatorial_radius
    r2 = apoapsis + body.equatorial_radius
    a = (r1 + r2) / 2
    v1 = visviva(body, r1, a)
    v2 = visviva(body, r2, a)
    return v1, v2

def orbital_period(body, a):
    return 2* np.pi * np.sqrt(a**3 / body.standard_gravitational_parameter)

def semimajoraxis(body,r1,r2):
    x1 = body.equatorial_radius + r1
    x2 = body.equatorial_radius + r2
    return (x1+x2)/2

def grav_force(body, pos):
    mu = body.standard_gravitational_parameter
    r = vc.mag(pos)
    return -vc.normalize(pos) * mu / r**2

def closest_item(arr, n):
    greater_than_n = [x for x in arr if x > n]
    less_than_n = [x for x in arr if x < n]
    if greater_than_n:
        return min(greater_than_n, key=lambda x: x - n)
    if less_than_n:
        return max(less_than_n, key=lambda x: n - x)
    return None

def find_list_containing_number(arr, n):
    for sublist in arr:
        if n in sublist:
            return sublist
    return None

def remove_leading_trailing_zero_arrays(array_of_arrays):
    non_zero_indices = [i for i, arr in enumerate(array_of_arrays) if not np.all(arr == 0)]
    if not non_zero_indices:
        return []
    start_index = non_zero_indices[0]
    end_index = non_zero_indices[-1]
    return array_of_arrays[start_index:end_index+1].tolist()

# Mun Shape
ang = np.linspace(0,2*np.pi,3600)
mun_x = mun.equatorial_radius * np.cos(ang)
mun_y = mun.equatorial_radius * np.sin(ang)

# Initialization
periapsis = 7000
apoapsis  = 80000
semi_major = semimajoraxis(mun,periapsis,apoapsis)
targ_alt = 3000

r_i = np.array([mun.equatorial_radius+periapsis,0])
v_i = np.array([0,apside_velocity(mun,periapsis,apoapsis)[0]])

# Time Setting
dt = 0.1
t = np.arange(0,1000+dt,dt)

# Acceleration Initialization

d_th = 0.05
max_th = 10
thrust = np.arange(0,max_th,d_th)
param_list = np.zeros((len(thrust),6))

# Iterative simulation Loop
r_t = r_i
v_t = v_i

start_time = time.time()
for j,th in enumerate(thrust):
    for i in range(len(t)):
        if i == 0:
            r_t = r_i
            v_t = v_i
        else: 
            a_g = grav_force(mun,r_t)
            a_c = -th * vc.normalize(v_t)

            v_t = v_t + (a_g + a_c) * dt
            r_t = r_t + v_t * dt

            alt = vc.mag(r_t)  - mun.equatorial_radius
            ver_speed = -vc.sign(vc.vdot(v_t,a_g))*vc.mag(vc.vprj(v_t,a_g))
            hor_speed =  vc.mag(vc.vxcl(v_t,a_g))

            if hor_speed < 30 or alt>7500:
                break
            if alt < targ_alt:
                drn_ang  = np.rad2deg(np.arctan(r_t[1]/r_t[0]))
                dat_arr = np.array([th,t[i],hor_speed,ver_speed, alt, drn_ang])
                param_list[j] = dat_arr
                break
    progress_bar(th,max(thrust),start_time)
print()
param_list = remove_leading_trailing_zero_arrays(param_list)
forty_list, v_forty_list = [], []
for dat_arr in param_list:
    hor_spd = dat_arr[2]
    if hor_spd < 50 and hor_spd > 30:
        forty_list.append(dat_arr)
        v_forty_list.append(hor_spd)
smallest_v   = closest_item(v_forty_list,40)
final_thrust = find_list_containing_number(forty_list,smallest_v) 

print(final_thrust)

thrust  = final_thrust[0]
timer   = final_thrust[1]
drn_ang = final_thrust[5]

# Redoing Simulation for this specific thrust

dt = 0.05
t = np.arange(0,timer+dt,dt)

# Vector - Data List Creation
r = np.zeros((len(t),2))
v = np.zeros((len(t),2))
a = np.zeros((len(t),2))
alt = np.zeros((len(t),2))
speedlist = np.zeros((len(t),2))

# Simulation Loop
r[0] = r_i
v[0] = v_i
alt[0] = vc.mag(r_i) - mun.equatorial_radius
a_g = grav_force(mun,r_i)
speedlist[0] = np.array([vc.mag(vc.vxcl(v_i,a_g)), vc.mag(vc.vprj(v_i,a_g))])

start_time = time.time()
for i in range(1,len(t)):
    a_g = grav_force(mun,r[i-1])
    a_c = -thrust*vc.normalize(v[i-1])

    v[i] = v[i-1] + (a_g + a_c) * dt
    r[i] = r[i-1] + v[i] * dt

    alt[i] = vc.mag(r[i])  - mun.equatorial_radius

    ver_speed = -vc.sign(vc.vdot(v[i],a_g))*vc.mag(vc.vprj(v[i],a_g))
    hor_speed =  vc.mag(vc.vxcl(v[i],a_g))
    speedlist[i] = np.array([hor_speed,ver_speed])
    progress_bar(i,len(t),start_time)
print()

# Original Orbit
t_i = t

dt = 1
t = np.arange(0,orbital_period(mun,semi_major)+dt,dt)
r_i = np.array([mun.equatorial_radius+periapsis,0])
v_i = np.array([0,apside_velocity(mun,periapsis,apoapsis)[0]])

r_m = np.zeros((len(t),2))
v_m = np.zeros((len(t),2))
r_m[0] = r_i
v_m[0] = v_i

start_time = time.time()
for i in range(1,len(t)):
    a_g = grav_force(mun,r_m[i-1])
    v_m[i] = v_m[i-1] +  a_g * dt
    r_m[i] = r_m[i-1] + v_m[i] * dt
    progress_bar(i,len(t),start_time)
print()

# Plotting
plt.plot(mun_x,mun_y,color='grey')
plt.plot(r_m[:,0],r_m[:,1],color='royalblue',marker='')
plt.plot(r[:,0],r[:,1],color='red',marker='')

plt.plot(t_i,speedlist[:,1],color='black')
plt.plot(t_i,speedlist[:,0],color='blue')
plt.plot(t_i,alt,color='brown')

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.set_axisbelow(True)

plt.grid()
plt.show()

#we can use data from this one as baseline data for p63