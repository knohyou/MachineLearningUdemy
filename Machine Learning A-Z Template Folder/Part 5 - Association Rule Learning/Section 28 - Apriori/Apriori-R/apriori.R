# Apriori 

# Data Preprocessing
# Extract DAta
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
# Header = FALSE First row is not the header

# Apriori train using sparse matrix
# Have a product for each column 120 products
# rows correspond a 1 or 0 whether the product was purchased for that particular transaction

#install.packages('arules')
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', 
                            rm.duplicates = TRUE)
# Separator is a comma. Not the default for read.transactions

# How to handle duplilcate
# 5 duplicates with 1 transaction?
summary(dataset)
# Density proportion of non-zero values

# Frequency plot
itemFrequencyPlot(dataset, topN = 10)
# First 100 most purchased by customers

#################################3333
# Build Apriori Model
rules = apriori(data = dataset,
                parameter = list(support = 0.004, confidence = 0.2))
# What is miniminum support 
# Number of product purchased/ Total number of products
# Want to set product purchased 3 times a day over the week
# 4*7/7500
# Confidence set a certain value and decrease = 0.8
# Important to see how many rules created
# minimum confidence of 0.8. 80 percent of transaction need to have the rule

# Visualize the rules
# Sort with decreasing lift
inspect(sort(rules, by = 'lift')[1:10])
# Note: the baskets may contain products like chocolate or mineral water that are highly purchased. (High support) 






