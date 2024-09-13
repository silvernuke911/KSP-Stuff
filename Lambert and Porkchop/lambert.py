import numpy as np
import matplotlib.pyplot as plt
from ksp_planets import *
import vector_operations as vc


def a_g(M , G, r):
    a = (M * G) / ((vc.mag(r))**2) * vc.normalize(r)
    return -a

import numpy as np
from scipy.optimize import fsolve

def lambert_solver(r1, r2, tof, mu):
    def norm(v):
        return np.linalg.norm(v)

    def stumpff_c(z):
        if z > 0:
            return (1 - np.cos(np.sqrt(z))) / z
        elif z < 0:
            return (np.cosh(np.sqrt(-z)) - 1) / -z
        else:
            return 1/2

    def stumpff_s(z):
        if z > 0:
            return (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)**3)
        elif z < 0:
            return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z)**3)
        else:
            return 1/6

    r1_norm = norm(r1)
    r2_norm = norm(r2)
    cos_dnu = np.dot(r1, r2) / (r1_norm * r2_norm)
    sin_dnu = np.sqrt(1 - cos_dnu**2)
    A = sin_dnu * np.sqrt(r1_norm * r2_norm / (1 - cos_dnu))

    def time_of_flight(z):
        C = stumpff_c(z)
        S = stumpff_s(z)
        y = r1_norm + r2_norm + A * (z * S - 1) / np.sqrt(C)
        if y < 0:
            return float('inf')
        x = np.sqrt(y / C)
        return x**3 * S + A * np.sqrt(y)

    # Solve for short-path (direct) solution
    z_guess = 0.1
    z_solution_short = fsolve(lambda z: time_of_flight(z) - np.sqrt(mu) * tof, z_guess)[0]

    C_short = stumpff_c(z_solution_short)
    S_short = stumpff_s(z_solution_short)
    y_short = r1_norm + r2_norm + A * (z_solution_short * S_short - 1) / np.sqrt(C_short)
    f_short = 1 - y_short / r1_norm
    g_short = A * np.sqrt(y_short / mu)
    g_dot_short = 1 - y_short / r2_norm

    v1_short = (r2 - f_short * r1) / g_short
    v2_short = (g_dot_short * r2 - r1) / g_short

    # Solve for long-path (retrograde) solution
    z_guess = -0.1
    z_solution_long = fsolve(lambda z: time_of_flight(z) - np.sqrt(mu) * tof, z_guess)[0]

    C_long = stumpff_c(z_solution_long)
    S_long = stumpff_s(z_solution_long)
    y_long = r1_norm + r2_norm + A * (z_solution_long * S_long - 1) / np.sqrt(C_long)
    f_long = 1 - y_long / r1_norm
    g_long = A * np.sqrt(y_long / mu)
    g_dot_long = 1 - y_long / r2_norm

    v1_long = (r2 - f_long * r1) / g_long
    v2_long = (g_dot_long * r2 - r1) / g_long

    return (v1_short, v2_short), (v1_long, v2_long)


M = 1
G = 1

r1 = np.array([1,1])
r2 = np.array([-2,-4])

tof = 20

v_c1 = lambert_solver(r1,r2,tof,G*M)[0][0]
v_c2 = lambert_solver(r1,r2,tof,G*M)[0][1]

dt = 0.001
t = np.arange(0,tof+dt,dt)

p1 = np.zeros((len(t),2))
p2 = np.zeros((len(t),2))

v1 = np.zeros((len(t),2))
v2 = np.zeros((len(t),2))

p1[0] = r1
p2[0] = r2

v1[0] = v_c1
v2[0] = v_c2

for i in range(1,len(t)):
    a1 = a_g(M,G,p1[i-1])
    a2 = a_g(M,G,p2[i-1])

    v1[i] = v1[i-1] + a1 * dt
    v2[i] = v2[i-1] + a2 * dt

    p1[i] = p1[i-1] + v1[i] * dt
    p2[i] = p2[i-1] + v2[i] * dt

plt.scatter(r1[0],r1[1])
plt.scatter(r2[0],r2[1])
plt.scatter(0,0)

plt.plot(p1[:,0], p1[:,1], marker = '')
plt.plot(p2[:,0], p2[:,1], marker = '')

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.grid()
plt.show()

# Porkchop plot

# Pseudocode

# code something, Keostationary orbit, yes yes, data scraper for maps
# bounds =    tof  0.5 period of lower body and 1.5 period of higher body
#         departure time - now to period upper body (might have to revise, something something resonant orbits)
#         division  - 10000 divisions

# for i in time of flight:
#     for j in departure time 
#         prograde departure vector = lambert solver (position at departure, position at tof, GM)
#         mag = mag(prograde_departure vector)
#         record tof, dep time, mag
# plot the porkchop plot
# identify the lowest dv if departure time is anytime now
# identify lowest dv ever if most optimal time

# translate the velocity vector to a prograde/radial/normal vector
# done

# data scrape pseudo code


# // Define boundaries for latitude and longitude
# set start_lat to -90.
# set end_lat to -60.

# set start_long to 0.
# set end_long to 50.

# // Iterate over latitudes
# from { local lat is start_lat } until { lat >= end_lat } step { set lat to lat + 1/60 } {
#     // Iterate over longitudes
#     from { local long is start_long } until { long >= end_long } step { set long to long + 1/60 } {
#         // Obtain terrain height
#         local local_height is geoposition(lat, long):terrainheight.
#         // Log latitude, longitude, and terrain height to a CSV file
#         log lat + ',' + long + ',' + local_height to terrainheight.csv.
#     }
# }

# start_lat = -90
# end_lat   = -60

# start_long = 0
# end_long = 50

# step_size = 1/60
# lat_list = np.arange(start_lat, end_lat+step_size, step_size)
# long_list = np.arange(start_long, end_long+step_size, step_size)

# terrain_list = np.ndarray([0,0])

# def terrain_height(lat,long):
#     x = lat * long
#     pass
# def record(*args):
#     pass

# for lat in lat_list:
#     for long in long_list:
#         height = terrain_height(lat,long)
#         record (lat,long,height)

# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# bounds = [0.5, 1.5]  # Boundaries for time of flight
# departure_time_range = [0, 1]  # Boundaries for departure time
# divisions = 10000  # Number of divisions

# # Generate time of flight and departure time arrays
# tof_values = np.linspace(bounds[0], bounds[1], divisions)
# departure_time_values = np.linspace(departure_time_range[0], departure_time_range[1], divisions)

# # Arrays to store results
# dv_values = np.zeros((divisions, divisions))

# # Loop through time of flight and departure time values
# for i, tof in enumerate(tof_values):
#     for j, departure_time in enumerate(departure_time_values):
#         # Calculate departure vector
#         r1 = np.array([0, 0, 0])  # Assuming Earth at origin
#         r2 = np.array([1, 0, 0])  # Assuming target at (1, 0, 0) (normalized distance)
#         prograde_departure_vector, _ = lambert_solver(r1, r2, tof, G*M)
#         dv_values[i, j] = np.linalg.norm(prograde_departure_vector)

# # Plot the porkchop plot
# plt.figure(figsize=(10, 6))
# plt.contourf(departure_time_values, tof_values, dv_values, cmap='viridis')
# plt.colorbar(label='Delta-v (km/s)')
# plt.xlabel('Departure Time')
# plt.ylabel('Time of Flight')
# plt.title('Porkchop Plot')
# plt.grid(True)
# plt.show()

# # Find lowest delta-v if departure time is anytime now
# min_dv_now = np.min(dv_values[:, np.abs(departure_time_values - 0.5).argmin()])
# print("Lowest delta-v if departure time is anytime now:", min_dv_now)

# # Find lowest delta-v ever
# min_dv_ever = np.min(dv_values)
# min_dv_ever_indices = np.where(dv_values == min_dv_ever)
# optimal_tof_index = min_dv_ever_indices[0][0]
# optimal_departure_time_index = min_dv_ever_indices[1][0]
# optimal_tof = tof_values[optimal_tof_index]
# optimal_departure_time = departure_time_values[optimal_departure_time_index]
# print("Lowest delta-v ever:", min_dv_ever)
# print("Optimal Time of Flight:", optimal_tof)
# print("Optimal Departure Time:", optimal_departure_time)

#works ggood but needs checking

def orbital_elements_to_vectors(a, e, i, Omega, omega, nu, mu):
    """
    Convert classical orbital elements to position and velocity vectors.

    Parameters:
    a (float): Semi-major axis (in kilometers).
    e (float): Eccentricity.
    i (float): Inclination (in radians).
    Omega (float): Longitude of the ascending node (in radians).
    omega (float): Argument of periapsis (in radians).
    nu (float): True anomaly (in radians).
    mu (float): Gravitational parameter (default value for Earth in km^3/s^2).

    Returns:
    pos (numpy.ndarray): Position vector in the inertial frame (in kilometers).
    vel (numpy.ndarray): Velocity vector in the inertial frame (in kilometers/second).
    """
    # Compute the eccentric anomaly
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
    
    # Compute the radius and velocity magnitude
    r = a * (1 - e**2) / (1 + e * np.cos(nu))
    v = np.sqrt(mu * (2 / r - 1 / a))
    
    # Compute the position and velocity in the perifocal frame
    pos_perifocal = np.array([r * np.cos(E), r * np.sin(E), 0])
    vel_perifocal = np.array([-v * np.sin(E), v * (np.cos(E) - e), 0])
    
    # Compute the transformation matrix from perifocal to inertial frame
    R = np.array([
        [np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.sin(Omega) * np.cos(i),
         -np.cos(omega) * np.sin(Omega) - np.sin(omega) * np.cos(Omega) * np.cos(i),
         np.sin(omega) * np.sin(i)],
        [np.sin(omega) * np.cos(Omega) + np.cos(omega) * np.sin(Omega) * np.cos(i),
         -np.sin(omega) * np.sin(Omega) + np.cos(omega) * np.cos(Omega) * np.cos(i),
         -np.cos(omega) * np.sin(i)],
        [np.sin(omega) * np.sin(i), np.cos(omega) * np.sin(i), np.cos(i)]
    ])
    
    # Transform position and velocity from perifocal to inertial frame
    pos = np.dot(R, pos_perifocal)
    vel = np.dot(R, vel_perifocal)
    
    return pos, vel

print(orbital_elements_to_vectors(1.2,0.5,60,120,30,40,1)) 