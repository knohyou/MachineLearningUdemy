# Thompson Sampling

# Import the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Import data
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implement UCB
# Initialize
import random
N = 10000
d = 10 
numbers_of_rewards_1 = d*[0]
numbers_of_rewards_0 = d*[0]

ads_selected = []
total_reward = 0

# Compute 
for n in range(0,N):
    max_random = 0
    ad = 0
    for i in range(0,d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else: 
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
        
    total_reward = total_reward + reward
    
# Visualising the results
plt.hist(ads_selected)
plt.xlabel('Ad')
plt.ylabel('Number of times selected')
plt.title('Histogram of Ad Selection')
plt.show()

            