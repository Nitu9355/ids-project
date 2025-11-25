# Load dataset


df <- read.csv("heart.csv", stringsAsFactors = FALSE)



head(df)



cat("Rows:", nrow(df), "\nColumns:", ncol(df))



dim(df)



str(df)


summary(df)



get_mode <- function(x) {
  uniq <- unique(x)
  uniq[which.max(tabulate(match(x, uniq)))]
}

sapply(df, get_mode) 




numeric_features <- names(df)[sapply(df, is.numeric)]
numeric_features 




categorical_features <- names(df)[sapply(df, function(x) is.factor(x) || is.character(x))]
categorical_features



# complete part A 




install.packages("ggplot2")   # run once
library(ggplot2)              # run every session


numeric_cols <- names(df)[sapply(df, is.numeric)]

for (col in numeric_cols) {
  print(
    ggplot(df, aes_string(col)) +
      geom_histogram(bins = 30) +
      ggtitle(paste("Histogram of", col))
  )
} 




# boxplot 


for (col in numeric_cols) {
  print(
    ggplot(df, aes_string(y = col)) +
      geom_boxplot() +
      ggtitle(paste("Boxplot of", col))
  )
}






# bar chart


categorical_cols <- names(df)[sapply(df, function(x) is.character(x) | is.factor(x))]


#frequency of categorical variable
for (col in categorical_cols) {
  print(
    ggplot(df, aes_string(col)) +
      geom_bar() +
      ggtitle(paste("Frequency of", col))
  )
}



# Correlation Matrix (Heatmap) 
install.packages("corrplot")
library(corrplot)


num_data <- df[numeric_cols]
cor_matrix <- cor(num_data, use = "complete.obs")

corrplot(cor_matrix, method = "color", type = "upper")




# scatter

pairs(num_data)



#Boxplots Between Categorical & Numerical Features

if ("target" %in% names(df)) {
  for (col in numeric_cols) {
    print(
      ggplot(df, aes_string(x = "target", y = col)) +
        geom_boxplot() +
        ggtitle(paste("Target vs", col))
    )
  }
}


# Skewness (Check Data Distribution Shape)
install.packages("reshape2")

library(reshape2)

install.packages("moments")
library(moments)


install.packages("e1071")
library(e1071)





df <- read.csv("heart.csv", stringsAsFactors = FALSE)




numeric_cols <- names(df)[sapply(df, is.numeric)]
numeric_cols



num_data <- df[numeric_cols]

head(num_data)





sapply(num_data, skewness)





# 📌 3.2 Count Outliers Per Column (IQR Method)




outlier_summary <- sapply(num_data, function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  sum(x < (Q1 - 1.5 * IQR_val) | x > (Q3 + 1.5 * IQR_val))
})

outlier_summary





# Part B complete




# 1. HANDLING MISSING VALUES

colSums(is.na(df))


# 📌 1.2 Replace or remove missing values

# Fill numeric columns with median
num_cols <- names(df)[sapply(df, is.numeric)]
df[num_cols] <- lapply(df[num_cols], function(x){
  x[is.na(x)] <- median(x, na.rm = TRUE)
  return(x)
})

# Fill categorical columns with mode
get_mode <- function(x) {
  uniq <- unique(x)
  uniq[which.max(tabulate(match(x, uniq)))]
}

cat_cols <- names(df)[sapply(df, is.character)]
df[cat_cols] <- lapply(df[cat_cols], function(x){
  x[is.na(x)] <- get_mode(x)
  return(x)
})





# 2. HANDLING OUTLIERS

# 2.1 Identify outliers (IQR method)


outlier_summary <- sapply(df[num_cols], function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR_val <- Q3 - Q1
  sum(x < (Q1 - 1.5 * IQR_val) | x > (Q3 + 1.5 * IQR_val))
})

outlier_summary



# 2.2 Remove or cap outliers


df_out <- df

for (col in num_cols) {
  Q1 <- quantile(df_out[[col]], 0.25)
  Q3 <- quantile(df_out[[col]], 0.75)
  IQR_val <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR_val
  upper <- Q3 + 1.5 * IQR_val
  
  df_out[[col]][df_out[[col]] < lower] <- lower
  df_out[[col]][df_out[[col]] > upper] <- upper
}


# ⭐ 3. DATA CONVERSION (ENCODING) 


# Convert categorical variables to factors

df_out[cat_cols] <- lapply(df_out[cat_cols], as.factor)



# One-hot encoding using model.matrix() 

# Identify columns with only one unique value
one_level_cols <- names(df_out)[sapply(df_out, function(x) length(unique(x)) == 1)]

# Print them
print(one_level_cols)

# Remove them
df_out <- df_out[, !(names(df_out) %in% one_level_cols)]


df_encoded <- model.matrix(~ . - 1, data = df_out)
df_encoded <- as.data.frame(df_encoded)



# 4. DATA TRANSFORMATION (SCALING)


# 4.1 Normalize (Min-Max scaling) 

df_norm <- as.data.frame(scale(df_encoded, center = FALSE, scale = apply(df_encoded, 2, max)))


# 4.2 Standardize (Z-score scaling) 


df_scaled <- as.data.frame(scale(df_encoded))


# 4.3 Fix skewness using log transformn 


skew_vals <- sapply(df_encoded, function(x) if(is.numeric(x)) e1071::skewness(x) else NA)
skewed_cols <- names(skew_vals[abs(skew_vals) > 1])

df_log <- df_encoded
for (col in skewed_cols) {
  df_log[[col]] <- log(df_log[[col]] + 1)
}



# 5. FEATURE SELECTION 

# 5.1 Correlation Analysis 

install.packages("corrplot")
library(corrplot)


cor_matrix <- cor(df_encoded[, sapply(df_encoded, is.numeric)])

corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.cex = 0.8,
         number.cex = 0.6)




#. 5.2 Remove features with low variance

install.packages("caret")

install.packages(c("ggplot2", "lattice"))
install.packages("caret")


library(caret)
nzv <- nearZeroVar(df_encoded)
df_selected <- df_encoded[, -nzv]




# 5.3 Use random forest importance 

install.packages("randomForest")

library(randomForest)

set.seed(123)
rf_model <- randomForest(df_selected[, -1], df_selected[, 1], importance = TRUE)

importance(rf_model)
varImpPlot(rf_model)




# complete project















