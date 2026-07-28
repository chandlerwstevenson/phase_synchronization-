import numpy as np  

class PilotMeasuremnt:  
    def __init__(self, R): 
        self.R = R  

    def measure(self, true_state): 
        phi = true_state[0]
        omega = true_state[1] 

        z = np.array([
            np.cos(phi), 
            np.sin(phi), 
            omega 
        ])

        noise = np.random.multivariate_normal(
            np.zeros(3), 
            self.R
        )

        return z + noise  
    
