# Apriori

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Apriori expecting a list as input
# list of list

# Loop of all products and all of transactions
transactions = []
for i in range(0,7501): # i takes 0 to 7500 
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    # Need brackets to make it into a list
    # Need to products as a string

# Training the Apriori 
# Directory has a library file directly import
from apyori import apriori
rules = apriori(transactions, 
                min_support = 0.003, # Items purchased 3 times per day 
                min_confidence = 0.2,
                min_lift = 3, # Lift greater than 3 is good relevant rules
                min_length = 2)

# Visualising the results
results = list(rules) # rules already sorted by its criterion



# This function takes as argument your results list and return a tuple list with the format:
# [(rh, lh, support, confidence, lift)] 
def inspect(results):
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))
 
# this command creates a data frame to view
resultDataFrame=pd.DataFrame(inspect(results))
