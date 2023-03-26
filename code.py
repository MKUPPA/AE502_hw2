"""
Gauss's Planetary Equations 
for J2 perturbation
"""
import math
import utils as ul
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Constants
J2 = 0.00108 # J2 perturbation
R = 6370 # Earth's radius in km
GM = 3.986E5 # Earth's gravitational parameter in km^3/s^2

def odes(x, t):
    
    """
    The ode set corresponding to Gauss's planetary equations are
    taken from Curtis, 4th edition, Ch 12, (12.89a) - (12.89f)
    
    x holds:
    - Angular momentum, h
    - eccentricity, e
    - true anomaly, theta
    - argument of ascending node, ohm
    - inclination, i
    - argument of perigee, omega
    
    """
    
    # constants
    J2 = 0.00108 # J2 perturbation
    R = 6370 # Earth's radius in km
    GM = 3.986E5 # Earth's gravitational parameter in km^3/s^2
    const1 = -1.5*J2*GM*R**2 # constant used in odes
    
    # assign each ode to a vector element
    h = x[0]
    e = x[1]
    theta = x[2]
    ohm = x[3]
    i = x[4]
    omega = x[5]
    
    # define each ode
    r = (h**2)/(GM*(1 + e*np.cos(theta)))
    u = omega + theta
    
    # Angular momentum
    dh_dt = (const1/r**3)*np.sin(i)**2*np.sin(2*u)
    
    # Eccentricity
    term1 = (h**2/(GM*r))*np.sin(theta)*(3*np.sin(i)**2*np.sin(u)**2 - 1)
    term2 = np.sin(2*u)*np.sin(i)**2*(np.cos(theta)*(2 + e*np.cos(theta)) + e)
    de_dt = (-1*const1/(h*r**3))*(term1 - term2)
    
    # true anomaly
    term1 = (h**2/(GM*r))*np.cos(theta)*(3*np.sin(i)**2*np.sin(u)**2 - 1)
    term2 = (2 + e*np.cos(theta))*np.sin(2*u)*np.sin(i)**2*np.sin(theta) 
    dtheta_dt = (h/r**2) + (-1*const1/(e*h*r**3))*(term1 + term2)
    
    # argument of ascending node
    dohm_dt = (2*const1/(h*r**3))*np.sin(u)**2*np.cos(i)
    
    # inclination
    di_dt = (const1/(2*h*r**3))*np.sin(2*u)*np.sin(2*i)
    
    # argument of perigee
    term1 = (h**2/(GM*r))*np.cos(theta)*(1 - 3*np.sin(i)**2*np.sin(u)**2)
    term2 = (2 + e*np.cos(theta))*np.sin(2*u)*np.sin(i)**2*np.sin(theta)
    domega_dt = (-1*const1/(e*h*r**3))*(term1 - term2 + 2*e*np.cos(i)**2*np.sin(u)**2)
    
    return [dh_dt, de_dt, dtheta_dt, dohm_dt, di_dt, domega_dt]


# initial semimajor axis
a = 26600 #km

# initial inclination
i0 = 1.10654 #rad

# initial eccentricity
e0 = 0.74

# initial argument of perigee
omega_0 = 5 #deg
omega_0 = math.radians(omega_0)

# initial argument of ascending node
ohm_0 = 90 
ohm_0 = math.radians(ohm_0)

# initial mean anomaly
M0 = 10 #deg
M0 = math.radians(M0)

# compute initial angular momentum
h0 = np.sqrt(GM*a*(1 - e0**2)) # km^2/s

# compute initial true anomaly
theta_0 = ul.solve_kepler(M=M0, e=e0, tol=1e-7) # rad

# initial condition array
x0 = [h0, e0, theta_0, ohm_0, i0, omega_0]

# time array for 100 days --> convert to seconds
nb_days = 10
t = np.linspace(0, nb_days*24*60*60, 100000)
t_plot = t/(24*60*60)

x = odeint(odes, x0, t)


# Extract each quantity
h = x[:,0]
e = x[:,1]
theta = x[:,2]
ohm = x[:,3]
i = x[:,4]
omega = x[:,5]

# Plotting

plt.figure()
plt.plot(t_plot, h)
# plt.plot(t_plot, h - h[0])
plt.xlabel('Days')
plt.ylabel('Angular momentum (km^2/s)')
# plt.savefig('h_mean_100.pdf',dpi=600,bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(t_plot, e)
# plt.plot(t_plot, e - e[0])
plt.xlabel('Days')
plt.ylabel('Eccentricity')
# plt.savefig('e_mean_100.pdf',dpi=600,bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(t_plot, ohm*180/np.pi)
# plt.plot(t_plot, (ohm - ohm[0])*180/np.pi)
plt.xlabel('Days')
plt.ylabel('Argument of ascending node (deg)')
# plt.savefig('RAAN_mean_100.pdf',dpi=600,bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(t_plot, i*180/np.pi)
# plt.plot(t_plot, (i-i[0])*180/np.pi)
plt.xlabel('Days')
plt.ylabel('Inclination (deg)')
# plt.savefig('i_mean_100.pdf',dpi=600,bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(t_plot, omega*180/np.pi)
# plt.plot(t_plot, (omega-omega[0])*180/np.pi)
plt.xlabel('Days')
plt.ylabel('Argument of perigee (deg)')
# plt.savefig('w_mean_100.pdf',dpi=600,bbox_inches='tight')
plt.show()