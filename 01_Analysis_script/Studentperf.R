
# ==========================================================
# STUDENT PERFORMANCE ANALYSIS
# Author: Arouna Romeo KONE
# ==========================================================


# ==========================================================
# 0. SETUP
# ==========================================================

# Load necessary packages

packages <- c("dplyr", "ggplot2", "tidyr", "readr", "stringr", "GGally", "caret", "corrplot","randomForest","xgboost","factoextra","Metrics","yardstick", "pROC","knitr","kableExtra")
installed <- packages %in% installed.packages()
if(any(!installed)) install.packages(packages[!installed])
lapply(packages,library, character.only= TRUE)

# Load dataset
data <- read.csv("/Users/konearounaromeo/Downloads/student-scores.csv")


# ==========================================================
# 1. DEFINE THE OBJECTIVE
# ==========================================================
# 🎯 Goal: Understand what factors impact student academic performance
# 🎯 Target variables: math_score, physics_score, bilogy_score, geography_score...
# 🎯 Questions:
#   - Are absences related to lower scores?
#   - Do self-study hours help?
#   - Are there differences by gender, job status, or extracurriculars?

# ==========================================================
# 2. DATA CLEANING & PREPROCESSING
# ==========================================================

# Drop non-useful personal columns
data <- data %>%
  select(-id, -first_name, -last_name, -email)

# Check missing values
colSums(is.na(data))

# Convert categorial variables to factors 

categorial_vars <- c('gender','part_time_job','extracurricular_activities','career_aspiration')
data[categorial_vars] <- lapply(data[categorial_vars], as.factor)

lapply(data[categorial_vars], levels)

# Convert to long format for faceting
data_long <- pivot_longer(
  data[categorial_vars], 
  cols = everything(), 
  names_to = "variable", 
  values_to = "level"
)

# Plot all variables in one grid
ggplot(data_long, aes(x = level)) +
  geom_bar(fill = "steelblue") +
  facet_wrap(~variable, scales = "free") +
  labs(title = "Distribution of Categorical Variables") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-labels



# ==========================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================


# Overview of the data
names(data)
glimpse(data)
summary(data)

# Distribution of scores
score_vars <- names(data)[grepl('_score',names(data))]

for (var in score_vars) {
  p <- ggplot(data, aes(x = .data[[var]])) +
    geom_histogram(
      aes(y = after_stat(density)),bins = 20,fill = "steelblue",color = 'white') +
    geom_density(color = 'red', linewidth = 1) +
    theme_minimal() +
    ggtitle(paste('Distribution of:', var))
  
  print(p)
}

# Count gender 
gender_count <- data %>% count(gender)

# Gender-based average scores
table_mean <- data %>%
  group_by(gender) %>%
  summarise(across(all_of(score_vars), mean, na.rm = TRUE))
View(table_mean)

# Correlation between scores 

score_data <- data %>% select(all_of(score_vars))
corrplot(cor(score_data),method = "color",addCoef.col = "black",
col = colorRampPalette(c("darkred", "white", "darkgreen"))(100),
tl.cex = 0.8,
number.cex = 0.6,
mar = c(0, 0, 1, 0))  # Réduit les marges

# * throught the corrplot there is no multicolinearity between the scores*#

# Bivariate analysis 
# Build average score for all score vars 

data <- data %>% mutate(average_score = rowMeans(select(.,all_of(score_vars))))

# Correlation with average score
data_corr <- cor(data[, c(score_vars, "average_score")])
cor_average <- data_corr["average_score", score_vars]
cor_average <- sort(abs(cor_average), decreasing = TRUE)
print(cor_average)
barplot(cor_average, las = 2, col = "tomato",
        main = "Correlation with average score")


# * Mathematics and biology scores are strongly corelated with the average score 

# Relationship between explanatory variables and performance

# Study time
ggplot(data, aes(x = weekly_self_study_hours, y = average_score)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", color = "darkred") +
  theme_minimal() +
  labs(title = "Study time and average score")

# Part-time job
ggplot(data, aes(x = part_time_job, y = average_score)) +
  geom_boxplot(fill = "lightblue") + theme_minimal() +
  labs(title = "Impact of part-time job", x = "Job", y = "Average score")

#* Those wo don't work at part time have better results (average score 82 vs 77)

# Extracurricular activities
ggplot(data, aes(x = extracurricular_activities, y = average_score)) +
  geom_boxplot(fill = "lightgreen") + theme_minimal() +
  labs(title = "Extracurricular activities vs score")

# Absences
ggplot(data, aes(x = absence_days, y = average_score)) +
  geom_point(alpha = 0.6) + geom_smooth(method = "lm", color = "purple") +
  theme_minimal() + labs(title = "Absences vs performance")

# Career aspiration
library(forcats)
top_careers <- data %>% count(career_aspiration) %>% top_n(5, n) %>% pull(career_aspiration)
data %>%
  filter(career_aspiration %in% top_careers) %>%
  mutate(career_aspiration = fct_infreq(career_aspiration)) %>%
  ggplot(aes(x = career_aspiration, y = average_score)) +
  geom_boxplot(fill = "coral") + theme_minimal() +
  labs(title = "Average score by career aspiration") + coord_flip()


# -------------------------
# 4. FEATURE ENGINEERING
# -------------------------
data <- data %>% mutate(
  high_achiever = as.factor(ifelse(average_score >= 85, 1, 0)),
  study_efficiency = average_score / (weekly_self_study_hours + 1) # The higher the study_efficiency score, the better the student's academic performance relative to their study time—indicating strong learning effectiveness.
)

View(data)

# -------------------------
# 5. TRAIN/TEST SPLIT
# -------------------------
# To properly evaluate each model on unseen data

set.seed(42)

# Data spliting 
trainIndex <- createDataPartition(data$average_score,p=0.8,list=FALSE)
train <- data[trainIndex,]
test <- data[-trainIndex,]
View(train)


# Display sizes of train and test sets
message("Training set size (rows, columns): ", paste(dim(train), collapse = ", "))  
message("Test set size (rows, columns): ", paste(dim(test), collapse = ", "))  
      



# -------------------------
# 6. MODELING - REGRESSION
# -------------------------
#  Goal: Predict continuous variable (average_score)

# 6.1 Linear regression (baseline)
model_lm <- lm(average_score ~ gender + part_time_job + absence_days + weekly_self_study_hours + extracurricular_activities, data = train)
pred_lm <- predict(model_lm, test)
lm_metrics <- postResample(pred_lm, test$average_score)
lm_metrics

# With an R² of 28.8%, your linear regression model explains only a small portion of the variation in students' average scores (0-100 scale).
#The RMSE of 5.30 and MAE of 4.30 indicate that, on average, the model's predictions deviate by ~5 points from the actual scores, suggesting limited predictive accuracy—though this might still be meaningful in an educational context where many factors influence grades.

# 6.2 Random Forest (robust non-linear model)
ctrl <- trainControl(method = "cv", number = 5)
tuned_rf <- train(
  average_score ~ .,
  data = train,
  method = "rf",
  trControl = ctrl,
  tuneLength = 5
)
pred_rf <- predict(tuned_rf, test)
rf_metrics <- postResample(pred_rf, test$average_score)
rf_metrics

# With an RMSE of 1.74 and R² of 94%, 
#our Random Forest model predicts scores with an average error below 2 points, explaining nearly all variability in the data, indicating outstanding performance.


# 6.3 XGBoost with cross-validation
train_matrix <- model.matrix(average_score ~ .  - career_aspiration, data = train)
test_matrix <- model.matrix(average_score ~ . - career_aspiration, data = test)
dtrain <- xgb.DMatrix(data = train_matrix, label = train$average_score)
dtest <- xgb.DMatrix(data = test_matrix, label = test$average_score)

xgb_cv <- xgb.cv(
  data = dtrain,
  nrounds = 100,
  objective = "reg:squarederror",
  nfold = 5,
  metrics = "rmse",
  verbose = 0
)

model_xgb <- xgboost(data = dtrain, nrounds = which.min(xgb_cv$evaluation_log$test_rmse_mean), objective = "reg:squarederror", verbose = 0)
pred_xgb <- predict(model_xgb, dtest)
xgb_metrics <- postResample(pred_xgb, test$average_score)
xgb_metrics


# With an RMSE of 1.28 and R² of 95.9%, your XGBoost model predicts scores with an average error of ~1.3 points, capturing nearly all data variability, indicating exceptional performance.

  
# -------------------------
# 7. MODELING - CLASSIFICATION
# -------------------------
# 📌 Goal: Predict high achievers (high_achiever)

# 7.1 Logistic regression
model_logit <- glm(high_achiever ~ gender + part_time_job + absence_days + weekly_self_study_hours + extracurricular_activities, data = train, family = "binomial")
pred_logit <- predict(model_logit, test, type = "response")
class_logit <- ifelse(pred_logit > 0.5, 1, 0)
conf_matrix_logit <- confusionMatrix(as.factor(class_logit), test$high_achiever)
conf_matrix_logit

# 7.2 Random Forest - Classification
model_rf_class <- randomForest(high_achiever ~ ., data = train)
pred_rf_class <- predict(model_rf_class, test)
conf_matrix_rf <- confusionMatrix(pred_rf_class, test$high_achiever)
varImpPlot(model_rf_class)
conf_matrix_rf

# 7.3 XGBoost classification
label_train <- as.numeric(as.character(train$high_achiever))
label_test <- as.numeric(as.character(test$high_achiever))
train_matrix_class <- model.matrix(high_achiever ~ . - average_score  - career_aspiration, data = train)
test_matrix_class <- model.matrix(high_achiever ~ . - average_score  - career_aspiration, data = test)
dtrain_class <- xgb.DMatrix(data = train_matrix_class, label = label_train)
dtest_class <- xgb.DMatrix(data = test_matrix_class, label = label_test)
model_xgb_class <- xgboost(data = dtrain_class, objective = "binary:logistic", nrounds = 50, verbose = 0)
pred_xgb_class <- predict(model_xgb_class, dtest_class)
xgb_class <- ifelse(pred_xgb_class > 0.5, 1, 0)
confusionMatrix(as.factor(xgb_class), test$high_achiever)

# Vérifier les variables avec un seul niveau
sapply(train, function(x) length(unique(x)))


# -------------------------
# 8. UNSUPERVISED CLUSTERING
# -------------------------
# 📌 Group students by similar profiles
cluster_data <- data %>% select(all_of(score_vars), weekly_self_study_hours, absence_days) %>% scale()

# Optimal cluster number (elbow, silhouette)
fviz_nbclust(cluster_data, kmeans, method = "wss")
fviz_nbclust(cluster_data, kmeans, method = "silhouette")

k_model <- kmeans(cluster_data, centers = 2)
fviz_cluster(k_model, data = cluster_data)

# Hierarchical clustering
hclust_model <- hclust(dist(cluster_data), method = "ward.D2")
plot(hclust_model, labels = FALSE, hang = -1, main = "Hierarchical Clustering")

# -------------------------
# 9. MODEL COMPARISON TABLE
# -------------------------
model_summary <- data.frame(
  Model = c("Linear Regression", "Random Forest", "XGBoost"),
  RMSE = c(lm_metrics["RMSE"], rf_metrics["RMSE"], xgb_metrics["RMSE"]),
  Rsquared = c(lm_metrics["Rsquared"], rf_metrics["Rsquared"], xgb_metrics["Rsquared"])
)
print(model_summary)

# ================================================
# 🧮 MODEL EVALUATION - REGRESSION & CLASSIFICATION
# ================================================

# -------------------------
# 📌 REGRESSION - EVALUATION
# -------------------------

# Calculate metrics for each regression model

# Fonction R² si Metrics ne l'a pas
R2 <- function(pred, actual) {
  1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
}

# Liste de tes prédictions
predictions <- list(
  "Linear Regression" = pred_lm,
  "Random Forest"     = pred_rf,
  "XGBoost"           = pred_xgb
)

# Calcul des métriques
regression_metrics <- data.frame(
  Model = names(predictions),
  RMSE  = sapply(predictions, function(p) Metrics::rmse(test$average_score, p)),
  MAE   = sapply(predictions, function(p) Metrics::mae(test$average_score, p)),
  R2    = sapply(predictions, function(p) R2(p, test$average_score))
)

regression_metrics

regression_metrics %>%
  kable(format = "html", digits = 3, caption = "Regression Metrics for Models") %>%
  kable_styling(full_width = FALSE, position = "center", bootstrap_options = c("striped", "hover", "condensed"))

# -----------------------------
# 📌 CLASSIFICATION - EVALUATION
# -----------------------------

# Helper functions for classification metrics
library(yardstick)
library(pROC)
library(dplyr)

test_class <- test

# Logistic regression
pred_logit_prob <- predict(model_logit, test_class, type = "response")
pred_logit_class <- factor(ifelse(pred_logit_prob > 0.5, 1, 0), levels = c(0, 1))

logit_df <- tibble(
  truth = factor(test_class$high_achiever, levels = c(0,1)),
  estimate = pred_logit_class
)

logit_accuracy <- yardstick::accuracy(logit_df, truth = truth, estimate = estimate)$.estimate
logit_precision <- yardstick::precision(logit_df, truth = truth, estimate = estimate)$.estimate
logit_recall <- yardstick::recall(logit_df, truth = truth, estimate = estimate)$.estimate
logit_f1 <- yardstick::f_meas(logit_df, truth = truth, estimate = estimate)$.estimate

roc_logit <- roc(test_class$high_achiever, pred_logit_prob)
auc_logit <- auc(roc_logit)

# Random Forest
pred_rf_class <- predict(model_rf_class, test_class)
prob_rf_class <- predict(model_rf_class, test_class, type = "prob")

rf_df <- tibble(
  truth = factor(test_class$high_achiever, levels = c(0,1)),
  estimate = pred_rf_class
)

rf_accuracy <- yardstick::accuracy(rf_df, truth = truth, estimate = estimate)$.estimate
rf_precision <- yardstick::precision(rf_df, truth = truth, estimate = estimate)$.estimate
rf_recall <- yardstick::recall(rf_df, truth = truth, estimate = estimate)$.estimate
rf_f1 <- yardstick::f_meas(rf_df, truth = truth, estimate = estimate)$.estimate

roc_rf <- roc(test_class$high_achiever, prob_rf_class[,2])
auc_rf <- auc(roc_rf)

# XGBoost
xgb_metrics_obj <- yardstick::metrics(bind_cols(truth = test_class$high_achiever, estimate = as.factor(xgb_class), prob = pred_xgb_class), truth, estimate, prob)
roc_xgb <- roc(as.numeric(test_class$high_achiever), pred_xgb_class)
auc_xgb <- auc(roc_xgb)
# Assurez-vous que xgb_class et pred_xgb_class sont définis correctement avant
xgb_df <- tibble(
  truth = factor(test_class$high_achiever, levels = c(0,1)),
  estimate = as.factor(xgb_class)
)

xgb_accuracy <- yardstick::accuracy(xgb_df, truth = truth, estimate = estimate)$.estimate
xgb_precision <- yardstick::precision(xgb_df, truth = truth, estimate = estimate)$.estimate
xgb_recall <- yardstick::recall(xgb_df, truth = truth, estimate = estimate)$.estimate
xgb_f1 <- yardstick::f_meas(xgb_df, truth = truth, estimate = estimate)$.estimate

roc_xgb <- roc(test_class$high_achiever, pred_xgb_class)
auc_xgb <- auc(roc_xgb)

# Compile results in a table
classification_metrics <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  Accuracy = c(logit_accuracy, rf_accuracy, xgb_accuracy),
  Precision = c(logit_precision, rf_precision, xgb_precision),
  Recall = c(logit_recall, rf_recall, xgb_recall),
  F1 = c(logit_f1, rf_f1, xgb_f1),
  AUC = c(auc_logit, auc_rf, auc_xgb)
)

print(classification_metrics)

install.packages(c("knitr","kableExtra"))
library(knitr)
library(kableExtra)

classification_metrics %>%
  kable(format = "html", digits = 3, caption = "Classification Metrics for Models") %>%
  kable_styling(full_width = FALSE, position = "center", bootstrap_options = c("striped", "hover", "condensed"))






