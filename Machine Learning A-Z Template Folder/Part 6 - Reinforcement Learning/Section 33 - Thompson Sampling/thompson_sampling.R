# Upper confidence Bound

# Import dataset

dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implement the UCB
d = 10 # Number of ads
N = 10000 # Total number of iteration

ads_selected = integer(0)

number_of_selections_1 = integer(d)
number_of_selections_0 = integer(d)
total_reward = 0

for (n in 1:N){
  max_random = 0
  ad = 0
  for (i in 1:d){
    random_beta = rbeta(n = 1,
                        shape1 = number_of_selections_1[i] + 1,
                        shape2 = number_of_selections_0[i] + 1)
    if (random_beta > max_random){
      max_random = random_beta
      ad = i
    } 
  }
  ads_selected = append(ads_selected, ad)
  reward = dataset[n,ad]
  if (reward == 1){
    number_of_selections_1[ad] = number_of_selections_1[ad] + 1
  }
  else {
    number_of_selections_0[ad] = number_of_selections_0[ad] + 1
  }
  
  total_reward = total_reward + reward
}

# Visualize the results
hist(ads_selected, 
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of Times each ad selected')
