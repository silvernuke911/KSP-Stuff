import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
from matplotlib.animation import PillowWriter
import matplotlib.font_manager as font_manager
import vector_operations as vc 
from progress import progress_bar

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'axes.formatter.use_mathtext': True,
    'font.size': 12
})
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')

def universal_lambert(r1, r2, tof, mu, t_m=1, psi=0, psi_u=4*np.pi**2, psi_l=-4*np.pi, max_iter=1000, tol=1e-10):
    def C2(z):
        if z > 0:
            return (1 - np.cos(np.sqrt(z))) / z
        elif z < 0:
            return (np.cosh(np.sqrt(-z)) - 1) / -z
        else:
            return 0.5
    def C3(z):
        if z > 0:
            return (np.sqrt(z) - np.sin(np.sqrt(z))) / (z**1.5)
        elif z < 0:
            return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (-z)**1.5
        else:
            return 1/6
    mag_r1 = np.linalg.norm(r1)
    mag_r2 = np.linalg.norm(r2)
    gamma = np.dot(r1, r2) / (mag_r1 * mag_r2)
    A = t_m * np.sqrt(mag_r1 * mag_r2 * (1 + gamma))
    if A == 0:
        print('An orbit cannot exist')
        return np.zeros(3), np.zeros(3)
    c2 = 0.5
    c3 = 1 / 6.0
    for _ in range(max_iter):
        B = mag_r1 + mag_r2 + A * (psi * c3 - 1) / np.sqrt(c2)
        if A > 0 and B < 0:
            psi_l += np.pi
            B = -B
        chi3 = (B / c2)**1.5
        tof_ = (chi3 * c3 + A * np.sqrt(B)) / np.sqrt(mu)
        if abs(tof - tof_) < tol:
            break
        if tof_ <= tof:
            psi_l = psi
        else:
            psi_u = psi
        psi = (psi_u + psi_l) / 2.0
        c2 = C2(psi)
        c3 = C3(psi)
    else:
        # print('Did not converge')
        return np.zeros(3), np.zeros(3)
    f = 1 - B / mag_r1
    g = A * np.sqrt(B / mu)
    g_dot = 1 - B / mag_r2
    f_dot = (f * g_dot - 1) / g
    v1 = (r2 - f * r1) / g
    v2 = f_dot * r1 + g_dot * v1
    return v1, v2

def compute_orbital_elements(r_i, v_i, G, M, t=0, t0=0):
    if len(r_i) == 2:
        r_i = np.append(r_i, 0)
        v_i = np.append(v_i, 0)
    
    # Gravitational parameter (mu = GM)
    mu = G * M
    
    # Specific angular momentum vector (h = r x v)
    h = np.cross(r_i, v_i)
    k = np.array([0, 0, 1])

    # Magnitude of specific angular momentum
    h_mag = np.linalg.norm(h)
    
    # Eccentricity vector (e = ((v x h) / mu) - (r / |r|))
    e = (np.cross(v_i, h) / mu) - (r_i / np.linalg.norm(r_i))
    
    # Magnitude of eccentricity vector
    e_mag = np.linalg.norm(e)
    
    # Semi-major axis (a)
    p = h_mag**2 / mu
    r_mag = np.linalg.norm(r_i)
    v_mag = np.linalg.norm(v_i)
    a = p / (1 - e_mag**2)
    
    # Orbital period
    T_p = 2 * np.pi * np.sqrt(a ** 3 / mu)
    
    # Inclination (i)
    i = np.arccos(h[2] / h_mag)
    
    # Node vector (n = k x h)
    n = np.cross(k, h)
  
    # Magnitude of node vector
    n_mag = np.linalg.norm(n)

    # Right ascension of the ascending node (RAAN, Ω)
    if n_mag != 0:
        Omega = np.arccos(n[0] / n_mag)
        if n[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0
    
    # Argument of periapsis (ω)
    if n_mag != 0 and e_mag != 0:
        omega = np.arccos(np.dot(n, e) / (n_mag * e_mag))
        if e[2] < 0:
            omega = 2 * np.pi - omega
    elif n_mag == 0 and e_mag != 0:
        omega = np.arccos(np.dot(np.array([1,0,0]),e)/(e_mag * 1))
        if np.cross(np.array([1,0,0]), e)[2] < 0 : 
            omega = 2 * np.pi - omega
    else:
        omega = 0
    
    print(e,r_i,np.dot(e, r_i),(e_mag * r_mag))
    print(np.arccos(1))
    # True anomaly (ν)
    if e_mag != 0:
        if np.dot(e, r_i) / (e_mag * r_mag) > 1 :
            nu = 0
        else: 
            nu = np.arccos(np.dot(e, r_i) / (e_mag * r_mag))
        if np.cross(r_i, v_i)[2] < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0
    
    # Eccentric anomaly (E)
    E = 2 * np.arctan(np.tan(nu / 2) * np.sqrt((1 - e_mag) / (1 + e_mag)))
    
    # Mean anomaly (M)
    M0 = (E - e_mag * np.sin(E)) - np.sqrt( G*M / a**3 ) * ( t - t0 )
    
    # Return orbital elements as a dictionary
    orbital_elements = {
        'semi_major_axis': a,
        'eccentricity': e_mag,
        'inclination': np.degrees(i),
        'LAN': np.degrees(Omega),
        'argument_of_periapsis': np.degrees(omega),
        'true_anomaly': np.degrees(nu),
        'orbital_period': T_p,
        'mean_anomaly_at_epoch': np.degrees(M0)
    }

    return orbital_elements

class State_vectors:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        orbital_parameters = compute_orbital_elements(position,velocity,1,1,0)
        self.semi_major_axis = orbital_parameters['semi_major_axis']
        self.eccentricity = orbital_parameters['eccentricity']
        self.inclination = orbital_parameters['inclination']
        self.LAN = orbital_parameters['LAN']
        self.argument_of_periapsis = orbital_parameters['argument_of_periapsis']
        self.true_anomaly = orbital_parameters['true_anomaly']
        self.orbital_period = orbital_parameters['orbital_period']
        self.mean_anomaly_at_epoch = orbital_parameters['mean_anomaly_at_epoch']

def solve_kepler(M, e):
    """
    Solves Kepler's equation M = E - e*sin(E) for E (Eccentric Anomaly)
    using the Newton-Raphson method.
    """
    E = M  # Initial guess
    tolerance = 1e-10
    max_iterations = 1000
    for iteration in range(max_iterations):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        E_new = E - f / f_prime
        if abs(E_new - E) < tolerance:
            return E_new
        E = E_new
    raise RuntimeError("Kepler's equation did not converge.")
def orbital_elements_to_state_vector(a, e, i, omega, w, M0, Mu, t0, t_f):
    """
    Given the orbital elements, computes the position and velocity vectors at a future time t_f.
    a: Semi-major axis (km)
    e: Eccentricity
    i: Inclination (radians)
    omega: Longitude of the ascending node (radians)
    w: Argument of periapsis (radians)
    M0: Mean anomaly at epoch t0 (radians)
    Mu: Standard gravitational parameter of main body
    t0: Initial time (epoch) (seconds)
    t_f: Future time (seconds)
    """

    # Mean motion (n) in radians per second
    n = np.sqrt(Mu / a**3)
    # Mean anomaly at time t_f
    M = M0 + n * (t_f - t0)
    # Solve Kepler's equation to get the Eccentric Anomaly (E)
    E = solve_kepler(M, e)
    # True anomaly (v)
    v = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    # Distance (r)
    r = a * (1 - e * np.cos(E))
    # Specific Angular momentum magnitude
    h = np.sqrt(Mu * a * (1 - e**2))
    # Position vector in orbital plane
    r_orb = r * np.array([np.cos(v), np.sin(v), 0])
    # Velocity vector in orbital plane
    v_orb = (Mu / h) * np.array((-np.sin(v), e + np.cos(v), 0))
    R3_omega = np.array([[np.cos(omega), -np.sin(omega), 0],
                         [np.sin(omega),  np.cos(omega), 0],
                         [            0,              0, 1]])
    R3_i = np.array([[1,         0,          0],
                     [0, np.cos(i), -np.sin(i)],
                     [0, np.sin(i),  np.cos(i)]])
    R3_w = np.array([[np.cos(w), -np.sin(w), 0],
                     [np.sin(w),  np.cos(w), 0],
                     [0,          0,         1]])
    # Combined rotation matrix
    R = R3_omega @ R3_i @ R3_w
    # Transform to inertial frame
    r_inertial = R @ r_orb
    v_inertial = R @ v_orb

    return r_inertial, v_inertial

def grav_force (m1,m2,r1,r2):
    G = 1
    r_c = (r2 - r1)
    return - G * m1 * m2 / vc.mag(r_c)**2  * vc.normalize(r_c)
M = 1
G = 1
O = np.array([0,0,0])

p1_i = np.array([1,0,0.01])
v1_i = np.array([0,1,0])
p2_i = np.array([2,0,0.1])
v2_i = np.array([0,0.707,0])
# p2_i = np.array([2,0])
# v2_i = np.array([0,0.6])

p_1 = State_vectors(0.5, p1_i ,v1_i)
p_2 = State_vectors(0.5, p2_i, v2_i)

print(compute_orbital_elements(p1_i,v1_i,G,M))
print(compute_orbital_elements(p2_i,v2_i,G,M))

dt = 0.005
max_t = p_2.orbital_period + p_1.orbital_period
t = np.arange(0,max_t+dt,dt)

r1 = np.zeros((len(t),3))
r2 = np.zeros((len(t),3))
v1 = np.zeros((len(t),3))
v2 = np.zeros((len(t),3))
a1 = np.zeros((len(t),3))
a2 = np.zeros((len(t),3))

r1[0] = p_1.position
r2[0] = p_2.position
v1[0] = p_1.velocity
v2[0] = p_2.velocity
a1[0] = grav_force(M,p_1.mass, O, r1[0]) / p_1.mass
a2[0] = grav_force(M,p_2.mass, O, r2[0]) / p_2.mass

time_start = time.time()
for i in range(1,len(t)):

    a1[i] = grav_force(M,p_1.mass, O, r1[i-1]) / p_1.mass
    a2[i] = grav_force(M,p_2.mass, O, r2[i-1]) / p_2.mass 

    v1[i] = v1[i-1] + a1[i] * dt
    v2[i] = v2[i-1] + a2[i] * dt

    r1[i] = r1[i-1] + v1[i] * dt
    r2[i] = r2[i-1] + v2[i] * dt

    progress_bar(i,len(t),time_start)
print()
print('Plotting')

delta_t = 0.1
departure_min = 0
departure_max = p_1.orbital_period * 1.5
departure_time = np.arange(departure_min, departure_max + delta_t, delta_t)

travel_min = 0
travel_max = p_2.orbital_period
travel_time =  np.arange(travel_min, travel_max + delta_t, delta_t)

state_list = np.zeros((len(departure_time)*len(travel_time),4))

# data plotting lmao
time_start = time.time()
k_total = len(departure_time)*len(travel_time)
k=0
for i, dep_time in enumerate(departure_time):
    for j, trv_time in enumerate(travel_time):
        p_1state = orbital_elements_to_state_vector(p_1.semi_major_axis,p_1.eccentricity,np.radians(p_1.inclination),np.radians(p_1.LAN),np.radians(p_1.argument_of_periapsis),np.radians(p_2.mean_anomaly_at_epoch),1,0,dep_time)
        p_2state = orbital_elements_to_state_vector(p_2.semi_major_axis,p_2.eccentricity,np.radians(p_2.inclination),np.radians(p_2.LAN),np.radians(p_2.argument_of_periapsis),np.radians(p_2.mean_anomaly_at_epoch),1,0,trv_time+dep_time)
        r_p1 = p_1state[0]
        r_p2 = p_2state[0]
        v_p1 = p_1state[1]
        v_p2 = p_2state[1]
        if np.cross(r_p1,r_p2)[2] > 0 :
            transfer_vectors = universal_lambert(r_p1,r_p2,trv_time,1,1)
            way_marker = 1
        else:
            transfer_vectors = universal_lambert(r_p1,r_p2,trv_time,1,-1)
            way_marker = -1
        deltav = vc.mag(transfer_vectors[0]-v_p1)
        if np.dot(v_p1,transfer_vectors[0]) < 0 :
            deltav = np.nan
        if vc.mag(transfer_vectors[0]) == 0 :
            deltav = np.nan
        if deltav > 2 :
            deltav = np.nan
        state_list[(i-1)*len(travel_time)+j] = np.array([dep_time,trv_time,deltav,way_marker])
        k+=1
        progress_bar(k,k_total,time_start)

print()
minimum_deltav = np.nanmin(state_list[:, 2])
print(minimum_deltav)
print(state_list)

for state in state_list:
    if minimum_deltav in state:
        final_state = state
print(final_state)

# Reshape data for contour plot
dep_time_grid, arr_time_grid = np.meshgrid(departure_time, travel_time)
delta_v_grid = state_list[:, 2].reshape(len(departure_time), len(travel_time))

# Plotting the porkchop plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(dep_time_grid, arr_time_grid, delta_v_grid.T, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.title('Porkchop Plot')
plt.xlabel('Departure Time')
plt.ylabel('Arrival Time')
plt.show()


t_i = final_state[0]
t_f = final_state[1] + t_i

print(t_i,t_f)
t_stepi = int(t_i / dt)
t_stepf = int(t_f / dt)

p_1state = orbital_elements_to_state_vector(p_1.semi_major_axis,p_1.eccentricity,np.radians(p_1.inclination),np.radians(p_1.LAN),np.radians(p_1.argument_of_periapsis),np.radians(p_1.mean_anomaly_at_epoch),1,0,t_i)
p_2state = orbital_elements_to_state_vector(p_2.semi_major_axis,p_2.eccentricity,np.radians(p_2.inclination),np.radians(p_2.LAN),np.radians(p_2.argument_of_periapsis),np.radians(p_2.mean_anomaly_at_epoch),1,0,t_f)
transfer_vector = universal_lambert(p_1state[0],p_2state[0],final_state[1],1,final_state[3])

t = np.arange(0,final_state[1]+dt,dt)
at = np.zeros((len(t),3))
rt = np.zeros((len(t),3))
vt = np.zeros((len(t),3))

at[0] =  grav_force(M,1, O, r1[t_stepi]) / 1
rt[0] = p_1state[0]
vt[0] = transfer_vector[0]

time_start = time.time()
for i in range(1,len(t)):
    at[i] = grav_force(M,1, O, rt[i-1]) / 1
    vt[i] = vt[i-1] + at[i] * dt
    rt[i] = rt[i-1] + vt[i] * dt

def plot():
    def set_axes_equal(x, y, z, ax):
        x_limits = [np.min(x), np.max(x)]
        y_limits = [np.min(y), np.max(y)]
        z_limits = [np.min(z), np.max(z)]
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        plot_radius = 0.6 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    print('Plotting')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(r1[:, 0], r1[:, 1], r1[:, 2], color='blue')
    ax.plot(r2[:, 0], r2[:, 1], r2[:, 2], color='red')
    ax.plot(rt[:, 0], rt[:, 1], rt[:, 2], color='green')

    ax.scatter(O[0], O[1], O[2], marker='o', color='yellow', zorder=2)

    ax.scatter(r1[0][0], r1[0][1], r1[0][2], marker='.', color='blue', zorder=2)
    ax.quiver(r1[0][0], r1[0][1], r1[0][2], v1[0][0], v1[0][1], v1[0][2], color='b')

    ax.scatter(rt[0][0], rt[0][1], rt[0][2], marker='.', color='blue', zorder=2)
    ax.quiver(rt[0][0], rt[0][1], rt[0][2], vt[0][0], vt[0][1], vt[0][2], color='g')

    ax.scatter(r2[0][0], r2[0][1], r2[0][2], marker='.', color='red', zorder=2)
    ax.quiver(r2[0][0], r2[0][1], r2[0][2], v2[0][0], v2[0][1], v2[0][2], color='r')

    ax.scatter(r1[t_stepi][0], r1[t_stepi][1], r1[t_stepi][2], marker='.', color='blue', zorder=2)
    ax.quiver(r1[t_stepi][0], r1[t_stepi][1], r1[t_stepi][2], v1[t_stepi][0], v1[t_stepi][1], v1[t_stepi][2], color='b')

    ax.scatter(r2[t_stepf][0], r2[t_stepf][1], r2[t_stepf][2], marker='.', color='red', zorder=2)
    ax.quiver(r2[t_stepf][0], r2[t_stepf][1], r2[t_stepf][2], v2[t_stepf][0], v2[t_stepf][1], v2[t_stepf][2], color='r')

    ax.scatter(p_1state[0][0], p_1state[0][1], p_1state[0][2], marker='.', color='g', zorder=2)
    ax.quiver(p_1state[0][0], p_1state[0][1], p_1state[0][2], p_1state[1][0], p_1state[1][1], p_1state[1][2], color='g')

    ax.scatter(p_2state[0][0], p_2state[0][1], p_2state[0][2], marker='.', color='g', zorder=2)
    ax.quiver(p_2state[0][0], p_2state[0][1], p_2state[0][2], p_2state[1][0], p_2state[1][1], p_2state[1][2], color='g')

    set_axes_equal(r2[:, 0], r2[:, 1], r2[:, 2], ax)
    ax.set_xlabel('$x$-axis (AU)')
    ax.set_ylabel('$y$-axis (AU)')
    ax.set_zlabel('$z$-axis (AU)')
    ax.set_title('Porkchop Plot 3D')

    plt.show()
    print('Plotted')
plot()


# PORKCHOP PLOTTING AND TRANSFER
#     OBTAIN BODY1 ORBITAL PARAMETERS    /\\
#     OBTAIN BODY2 ORBITAL PARAMETERS    /\\
#             OBTAIN BODY1 NOW POSITION  /\\
#             OBTAIN BODY2 TOF POSITION  /\\
#             OBTAIN BODY1 NOW VELOCITY  /\\
#             OBTAIN BODY2 TOF VELOCITY  /\\
#     LIMIT TIME NOW SEARCH SPACE        /\\
#         0 TO BODY2 ORBITAL PERIOD      /\\
#     LIMIT TOF SEARCH SPACE             /\\
#         BODY1 PERIOD / 2 TO BODY2 PERIOD  /\\
#     FOR TIME IN NOW SEARCH SPACE          /\\
#         FOR TIME IN TOF SEARCH SPACE      
#             OBTAIN BODY1 NOW POSITION     /\\
#             OBTAIN BODY2 TOF POSITION     /\\
#             OBTAIN BODY1 NOW VELOCITY     /\\
#             OBTAIN BODY2 TOF VELOCITY     /\\
#             OBTAIN V1 AND V2              /\\
#             OBTAIN DV1 AND DV2            /\\
#             RECORD NOW, TOF, DV1, DV2     /\\
#     TIME CONSTRAINT                          
#         ANY TIME NOW
#             FIND THE LOWEST DV WITHIN ONE ORBITAL PERIOD
#         LOWEST DV
#             FIND THE LOWEST DV IN THE ENTIRE SEARCH SPACE
#         RETURN DV1, TIME
#     INCLUDE APPROACH DV
#         MAP WITH ADDITION OF APPROACH DV, FIND THE LOWEST  
#     CREATE TRANSFER VECTOR
#     TRANSFORM TRANSFER VECTOR TO POLAR
#     EXECUTE NODE
