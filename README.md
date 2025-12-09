# Rate-The-Review
Authors: Chris Chen, Ishaan Tibdewal

## Overview
This is a data science project analyzing Food.com recipe data, where our focus is to predict recipe ratings from review text using TF-IDF and logistic regression. It also includes exploratory data analysis, missingness assessment, and hypothesis testing on recipe nutrition and user ratings from Food.com data.

---

## Introduction
Food connects people, cultures, and everyday experiences. Whether we’re trying a new recipe, sharing a meal with friends, or scrolling through online reviews to decide what to cook next, food is a huge part of how we explore, learn, and communicate. Recipe platforms like Food.com don’t just host ingredients and instructions, they capture thousands of real user experiences through reviews and ratings.

Both of us love cooking and discovering new dishes, and we noticed how often people rely on these reviews to decide what’s worth making. That sparked our interest in digging deeper: What actually makes a review helpful? Do certain ingredients or nutritional qualities influence ratings? Can we predict how well a recipe will be received just from the text people write?

To answer these questions, we analyzed two large datasets from Food.com—one containing over 80,000 recipes and another containing over 700,000 of user interactions. Our project explores this ecosystem from multiple angles: understanding missingness patterns, performing exploratory data analysis on recipe characteristics, testing hypotheses about nutrition and ratings, and building a predictive model using TF-IDF and logistic regression to classify review sentiment.

Through this process, we aim to better understand the relationship between how people talk about food and how they choose to rate it.

The first dataset that we examined, `recipes`, contains 83,782 rows and 12 columns with each row corresponding to an individual recipe. The 12 columns include:

| Column | Description |
| ----------- | ----------- |
| `name` | Recipe name|
| `id` | Recipe ID |
| `minutes`| Minutes to prepare recipe |
| `contributor_id` | User ID who submitted this recipe |
| `submitted`| Date recipe was submitted|
| `tags` | Food.com tags for recipe|
| `nutrition` | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for "percentage of daily value"|
| `n_steps` | Number of steps in the recipe|
| `steps` | Text for recipe steps, in order|
| `description`| User-provided description of the recipe|
| `ingredients` | Text for recipe ingredients|
| `n_ingredients` | Number of ingredients in the recipe|


The Second dataset that we examined, `interactions`, contains 731,927 rows and 5 columns with each row corresponding to an individual review from a user. The 5 



columns include :

| Column | Description |
| ----------- | ----------- |
| `user_id` | User ID who made the review|
| `recipe_id`| Recipe ID|
| `date`| Date of review|
| `rating`| Rating given on a 1-5 scale|
| `review`| Review text given|

