# Natural Language Processing

# importing the dataset



dataset_original = read.delim('Restaurant_Reviews.tsv',
                     quote = '', 
                     stringsAsFactors = FALSE) # Shouldn't analyze reviews as a factor (single entitity)
# Removed the quotes as empty

# Cleaning the texts
#install.packages('tm')
#install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus,content_transformer(tolower))
# as.character(corpus[[1]])
# Remove numbers 
corpus = tm_map(corpus,removeNumbers)
corpus = tm_map(corpus,removePunctuation)
corpus = tm_map(corpus,removeWords, stopwords())
corpus = tm_map(corpus,stemDocument)
corpus = tm_map(corpus,stripWhitespace)

# Create Bag of Words model
dtm = DocumentTermMatrix(corpus)
# Document of matrix Create sparse matrix
# Remove words that don't occur frequently
dtm = removeSparseTerms(dtm, 0.999)

# need to convert matrix to a dataframe
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked


# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting classifier to the Training set
# Create your classifier here
#install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)


# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
