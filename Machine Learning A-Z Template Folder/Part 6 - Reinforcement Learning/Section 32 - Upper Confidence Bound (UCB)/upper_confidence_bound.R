# Upper confidence Bound

# Import dataset

dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implement the UCB
d = 10 # Number of ads
N = 10000 # Total number of iteration

ads_selected = integer(0)

number_of_selections = integer(d)
sums_of_rewards = integer(d)
total_reward = 0

for (n in 1:N){
  max_upper_bound = 0
  ad = 0
  for (i in 1:d){
    if (number_of_selections[i]>0){
        average_reward = sums_of_rewards[i]/number_of_selections[i]
        delta_i = sqrt(3/2*log(n)/number_of_selections[i])
        upper_confidence = average_reward + delta_i
    } else {
         upper_confidence = 1e400
    }
    if (upper_confidence > max_upper_bound){
      max_upper_bound = upper_confidence
      ad = i
    } 
  }
  ads_selected = append(ads_selected, ad)
  number_of_selections[ad] = number_of_selections[ad] + 1
  reward = dataset[n,ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

# Visualize the results
hist(ads_selected, 
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of Times each ad selected')
  