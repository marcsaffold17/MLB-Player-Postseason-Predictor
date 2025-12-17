# Prediction of MLB Player Performance for the 2025 Season
# Marcus Saffold, Jack Fritschel

library(tidyverse)
library(dplyr)        
library(ggplot2)      
library(caret)        
library(randomForest)
mlb <- read.csv("both_seasons.csv")
regular <- read_csv("regular_season.csv")
post <- read_csv("postseason.csv")
view(mlb)
view(regular)
view(post)

# Post season vs regular season differential of continuous variables
reg_season <- mlb %>% select(reg_ba, reg_obp, reg_slug, reg_ops)
postseason <- mlb %>% select(post_ba, post_obp, post_slug, post_ops)
mlb_diff <- mlb %>%
  mutate(
    diff_obp = post_obp - reg_obp,
    diff_slug = post_slug - reg_slug,
    diff_ops = post_ops - reg_ops,
    diff_ba = post_ba - reg_ba
  )
summary(mlb_diff %>% select(diff_obp, diff_slug, diff_ops, diff_ba))

# Scatter plot of on-base percentage vs batting average
regular <- regular %>%
  mutate(season = "regular")
post <- post %>%
  mutate(season = "postseason")

both_split <- bind_rows(regular, post)

ggplot(both_split, aes(x = on_base_plus_slg, y = batting_average, color = season)) +
  geom_point() +
  labs(title = "On-Base Percentage vs Batting Average",
       x = "On-Base Percentage",
       y = "Batting Average") +
  theme_minimal()


# Correlation matrix
cor_matrix <- mlb %>%
  select(reg_ba, reg_obp, reg_slug, reg_ops, post_ba, post_obp, post_slug, post_ops) %>%
  cor()
print(cor_matrix)

# Data partition
predictors <- mlb %>%
  select(reg_hits, reg_hr, reg_walks, reg_strikeouts, reg_slug, reg_ba, reg_runs, reg_rbi)
target <- mlb$reg_obp
set.seed(123)
train_index <- createDataPartition(target, p = 0.7, list = FALSE)
X_train <- predictors[train_index, ]
X_test  <- predictors[-train_index, ]
y_train <- target[train_index]
y_test  <- target[-train_index]

# Linear regression model
lm_model <- lm(y_train ~ ., data = X_train)
lm_pred <- predict(lm_model, newdata = X_test)
summary(lm_model)

#random forest model
rf_model <- randomForest(x = X_train, y = y_train, ntree = 500, mtry = 4, importance = TRUE)
rf_pred <- predict(rf_model, newdata = X_test)
summary(rf_model)
importance(rf_model)
varImpPlot(rf_model)
eval_metrics <- function(pred, actual) {
  rmse <- sqrt(mean((pred - actual)^2))
  mae  <- mean(abs(pred - actual))
  r2   <- cor(pred, actual)^2
  return(list(RMSE = rmse, MAE = mae, R2 = r2))
}
eval_metrics(lm_pred, y_test)
eval_metrics(rf_pred, y_test)

# Walks only
lm_walks <- lm(y_train ~ reg_walks, data = X_train)
summary(lm_walks)
pred_walks <- predict(lm_walks, newdata = X_test)
MAE_walks <- mean(abs(pred_walks - y_test))

# Batting average only
lm_BA <- lm(y_train ~ reg_ba, data = X_train)
summary(lm_BA)
pred_BA <- predict(lm_BA, newdata = X_test)
MAE_BA <- mean(abs(pred_BA - y_test))

# Percent improvement
improvement <- ((MAE_BA - MAE_walks) / MAE_walks) * 100
print(paste("MAE using walks:", round(MAE_walks, 4)))
print(paste("MAE using batting average:", round(MAE_BA, 4)))
print(paste("Percent Improvement:", round(improvement, 2), "%"))