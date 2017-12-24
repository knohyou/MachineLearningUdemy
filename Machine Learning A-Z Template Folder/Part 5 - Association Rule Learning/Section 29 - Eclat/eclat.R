# Eclat

# Data Preprocessing
#install.packages('arules')
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', 
                            rm.duplicates = TRUE)

# How to handle duplilcate
# 5 duplicates with 1 transaction?
summary(dataset)
# Density proportion of non-zero values

# Frequency plot
itemFrequencyPlot(dataset, topN = 10)
# First 100 most purchased by customers

#################################3333
# Build Eclat Model
rules = eclat(data = dataset,
                parameter = list(support = 0.004, minlen = 2))
# only need support as a parameter
# set minlen = 2 so that we look at least 2 items purchased 
# Instead of lists, we obtain sets with eclat


# Visualize the rules
# Sort with decreasing support
inspect(sort(rules, by = 'support')[1:10])
# Note: the baskets may contain products like chocolate or mineral water that are highly purchased. (High support) 
