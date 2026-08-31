# Introduction to Kalman Filter
import numpy as np 
import matplotlib.pyplot as plt  

actual_range = 10
num_Measurements = 100
measured_range = np.random.normal(actual_range,0.1, num_Measurements)

plt.plot(measured_range, 'o-')
plt.show()

# Measurement Covariance 
R = 0.01 

C = .0001 # Process Covariance 

# current measurement 
x = np.zeros(num_Measurements)
# error covariance 
p = np.zeros(num_Measurements)

# previous measurement 
x_minus = np.zeros(num_Measurements) 
p_minus = np.zeros(num_Measurements)

# Kalman filter gain 
k_gain = np.zeros(num_Measurements)

x[0]= 0 
p[0]= 1 

for i in range(1,num_Measurements): 
    x_minus[i] = x[i-1] 
    p_minus[i] = p[i-1] + C   

    k_gain[i] = p_minus[i]/(p_minus[i]+R)
    x[i] = x_minus[i]+k_gain[i]*(measured_range[i]-x_minus[i])
    p[i] = (1 - k_gain[i])*p_minus[i]

plt.plot(x)
plt.plot(measured_range)
plt.axis([0,100, 8, 11])
plt.legend(x,'hi')
plt.show()



     




