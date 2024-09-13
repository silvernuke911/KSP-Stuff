import numpy as np
import matplotlib.pyplot as plt

m_i = 1000.0
m_dot = 0.5
force = 100.0
thrust = 1.0

dt = 0.05
max_t = 100
t = np.arange(0,max_t+dt,dt)

a = np.zeros_like(t)
v = np.zeros_like(t)
x = np.zeros_like(t)
m = np.zeros_like(t)

a_i = 0
v_i = 0
x_i = 0

a[0] = a_i
v[0] = v_i
x[0] = x_i
m[0] = m_i

for i in range(1,len(t)):
    m[i] = m[i-1] - m_dot*dt
    a[i] = force * thrust / m[i]
    v[i] = v[i-1] + a[i] * dt
    x[i] = x[i-1] + v[i] * dt

plt.plot(t,x)
plt.plot(t,v)
plt.plot(t,a)
plt.plot(t,m)

plt.grid()
plt.show()