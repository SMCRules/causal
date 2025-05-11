import pandas as pd
import numpy as np
np.random.seed(1234)

def g(size, p_small, u_1):
    """
    Assigns treatment A or B based on kidney stone size and a uniform random 
    draw u_1. Probabilities are normalized by prevalence p_small of each stone 
    size in the population.
    """
    # prob_small = 0.51
    if size == "small":
        prob_A = (87 / 700) / p_small
    else:  # size == "large"
        prob_A = (80 / 700) / (1 - p_small)

    return "A" if u_1 <= prob_A else "B"

def f(size, treatment, u_2):
    """
    Simulates recovery 1 or no recovery 0 conditional on  kidney stone size, 
    treatment "A" or "B", and a uniform random draw u_2.
    """
    if size == "small":
        p_recov = (81/87) if treatment == "A" else (234/270)
    else:
        p_recov = (192/263) if treatment == "A" else (50/80)
    
    return 1 if u_2 <= p_recov else 0

# Simulation parameters
patients_n = 10000
p_small = 0.51

# Simulation loop
sizes = []
treatments = []
recoveries = []

for patient in range(patients_n):
    u_0, u_1, u_2 = np.random.uniform(size=3)
    size = "small" if u_0 <= p_small else "large"
    treatment = g(size, p_small, u_1)
    recovery = f(size, treatment, u_2)
    sizes.append(size)
    treatments.append(treatment)
    recoveries.append(recovery)

# create data frame
kidney_data = pd.DataFrame({
    "size": sizes, 
    "treatment": treatments, 
    "recovery": recoveries
})

print(kidney_data.head())
print(kidney_data.groupby("treatment")["recovery"].mean())
