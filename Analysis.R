library(ggplot2)
library(GGally)
library(here)

# Load the data
data <- read.csv(here("Dataset", "secondary_data.csv"), sep = ";")
df <- data.frame(data)

head(df)

# Missing values
missing <- df[!complete.cases(df), ]
missing

# Summary
summary(df)

# Plots
ggpairs(df, lower = list(continuous = "smooth", combo = "facethist"),
    upper = list(continuous = "blank", combo = "box"), aes(color = class))

ggpairs(df, lower = list(continuous = "smooth", combo = "facethist"), upper = list(continuous = "blank", combo = "box"), aes(color = season))

ggpairs(df, lower = list(continuous = "smooth", combo = "facethist"), upper = list(continuous = "blank", combo = "box"), aes(color = habitat))
