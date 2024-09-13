import numpy as np
import matplotlib.pyplot as plt
# from ksp_planets import *
# import vector_operations as vc
# import time 
# from progress import progress_bar
# def execution_time(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         execution_seconds = end_time - start_time
#         print(f"Execution time of {func.__name__}: {execution_seconds:.6f} seconds")
#         return result
#     return wrapper

# Braeunig's solution, lets identify shortway first
# @execution_time
def lambert_solver(r1,r2,tof,mu):
    def vang(v1,v2):
        return np.arccos( np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    # Inputs : 
    #   r1 := first radius vector
    #   r2 := second radius vector
    #   tof := time of flight
    #   mu := gravitational parameter
    #   direction := 1 if prograde, -1 if retrograde

    #calculating magnitudes
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    # calculating difference in angles
    nu = vang(r1,r2)

    # calculating initialization values
    k = r1_mag * r2_mag * (1 - np.cos(nu))
    l = r1_mag + r2_mag
    m = r1_mag * r2_mag * (1 + np.cos(nu))

    print(k,l,m)
    # determining limits on p
    p_1 = k / (l + np.sqrt(2 * m)) 
    p_2 = k / (l - np.sqrt(2 * m)) 

    # trial p
    print(p_1)
    print(p_2)

    p = (p_1 + p_2) / 2


    def time_of_flight(p):
        a = m * k * p / ((2 * m - l**2) * p**2 + 2 * k * l * p - k**2)

        f = 1 - (r2_mag / p) * (1 - np.cos(nu))
        g = r1_mag * r2_mag * np.sin(nu) / np.sqrt(mu * p)

        # f_dot = np.sqrt(mu / p) * np.tan(nu / 2) * ( (1 - np.cos(nu)) / p - 1 / r1_mag - 1/r2_mag)

        cos_delta_E = 1 - (r1_mag / a) * (1 - f) 
        # sin_delta_E = - r1_mag * r2_mag * f_dot / np.sqrt( mu * a )

        if a > 0:
            delta_E = np.arccos(cos_delta_E)
            t = g + np.sqrt( a**3 / mu) * (delta_E - np.sin(delta_E))
        elif a < 0:
            delta_F = np.arccosh(cos_delta_E)
            t = g + np.sqrt( (-a)**3 / mu) * (np.sinh(delta_F) - delta_F)
        elif a == 0:
            t = np.inf 

        return t

    def optimize(p_min, p_max, tof, max_iter=1000, tol=1e-10):
        
        for i in range(max_iter):
            p = (p_max + p_min) / 2
            if time_of_flight(p) > tof:
                p_max = p
            else:
                p_min = p
            print(p_min, p, p_max, '\t' , time_of_flight(p),tof)
            if abs(tof - time_of_flight(p)) < tol:
                break
            if i+1 == max_iter:
                raise ValueError('Did not converge')
        return p

    p = optimize(1e20,p_1,tof)
    print(p)

    f = 1 - (r2_mag / p) * (1 - np.cos(nu))
    g = r1_mag * r2_mag * np.sin(nu) / np.sqrt(mu * p)
    
    f_dot = np.sqrt(mu / p) * np.tan(nu / 2) * ( (1 - np.cos(nu)) / p - 1 / r1_mag - 1/r2_mag)
    g_dot = 1 - (r1_mag / p ) * ( 1 - np.cos(nu))

    v1_s = ( r2 - f * r1 ) / g
    v2_s = f_dot * r1 + g_dot * v1_s

    # works for short way, so yey?

    # long way
    p = optimize(p_2/4 ,0,tof)
    print(p)

    f = 1 - (r2_mag / p) * (1 - np.cos(nu))
    g = r1_mag * r2_mag * np.sin(nu) / np.sqrt(mu * p)
    g_dot = 1 - (r1_mag / p ) * ( 1 - np.cos(nu))
    f_dot = np.sqrt(mu / p) * np.tan(nu / 2) * ( (1 - np.cos(nu)) / p - 1 / r1_mag - 1/r2_mag)

    v1_s = ( r2 - f * r1 ) / g
    v2_s = f_dot * r1 + g_dot * v1_s

    return [v1_s,v2_s]

# @execution_time
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
        print('Did not converge')
        return np.zeros(3), np.zeros(3)

    f = 1 - B / mag_r1
    g = A * np.sqrt(B / mu)
    g_dot = 1 - B / mag_r2
    f_dot = (f * g_dot - 1) / g

    v1 = (r2 - f * r1) / g
    v2 = f_dot * r1 + g_dot * v1

    return v1, v2

M = 1
G = 1

r1 = np.array([1, 0])
r2 = np.array([-1, 0.1])

tof = 5.14159
print(f'{tof = } \n')

# solution = universal_lambert(r1,r2,tof,G*M, -1)

v_c1 = universal_lambert(r1,r2,tof,G*M, 1)[0]
v_c2 = universal_lambert(r1,r2,tof,G*M, 1)[1]
v_c3 = universal_lambert(r1,r2,tof,G*M, -1)
print("\n", v_c1[0],v_c1[1])
print(v_c2[0],v_c2[1])
print(v_c3[0],v_c3[1])
dt = 0.001
t = np.arange(0,tof+dt,dt)

p1 = np.zeros((len(t),2))
p2 = np.zeros((len(t),2))

v1 = np.zeros((len(t),2))
v2 = np.zeros((len(t),2))

p1[0] = r1[:2]
p2[0] = r1[:2]

v1[0] = v_c1[:2]
v2[0] = v_c2[:2]

def normalize(v):
    return v / np.linalg.norm(v)

def a_g(M , G, r):
    a = (M * G) / ((np.linalg.norm(r))**2) * normalize(r)
    return -a

# start_time = time.time()
for i in range(1,len(t)):
    a1 = a_g(M,G,p1[i-1])
    a2 = a_g(M,G,p2[i-1])

    v1[i] = v1[i-1] + a1 * dt
    v2[i] = v2[i-1] + a2 * dt

    p1[i] = p1[i-1] + v1[i] * dt
    p2[i] = p2[i-1] + v2[i] * dt

    # progress_bar(i,len(t),start_time)
print()

plt.scatter(r1[0],r1[1], color = 'blue', zorder = 2)
plt.scatter(r2[0],r2[1], color = 'red', zorder = 2)
plt.scatter(0,0, marker = 'o', color = 'yellow', zorder = 3)

plt.plot(p1[:,0], p1[:,1], marker = '')
plt.plot(p2[:,0], p2[:,1], marker = '')

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.grid()
plt.show()

# # print(lambert_solver2(r1,r2,tof,mu))
