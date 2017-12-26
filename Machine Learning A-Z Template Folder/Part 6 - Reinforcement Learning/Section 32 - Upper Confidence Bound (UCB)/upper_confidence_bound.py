# Upper Confidence Bound

# Import the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Import data
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implement UCB
# Initialize
import math
N = 10000
d = 10 
numbers_of_selections = [0]*d
sums_of_rewards = [0]*d # Vector of size 1xd
ads_selected = []
total_reward = 0

# Compute 
for n in range(0,N):
    max_upper_bound = 0
    ad = 0
    for i in range(0,d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/numbers_of_selections[i]) # log is expecting first round 
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
            