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


The Second dataset that we examined, `interactions`, contains 731,927 rows and 5 columns with each row corresponding to an individual review from a user. The 5 columns include :

| Column | Description |
| ----------- | ----------- |
| `user_id` | User ID who made the review|
| `recipe_id`| Recipe ID|
| `date`| Date of review|
| `rating`| Rating given on a 1-5 scale|
| `review`| Review text given|


---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning
Before we dive right into exploring and working with our data, we first had to perform a few cleaning steps to prepare our datasets for better and more efficient analysis.

1. We first mereged our `recipes` dataset with the `interactions` dataset on `id` and `recipe_id`, respecitvely.
   - The resulting dataset has a row for each unique review corresponding to the recipe. The resulting dataframe is `recipes_interactions`.
2. We then replaced all ratings of 0 with `np.nan`.
   - This makes sense for our dataset since ratings are only a scale of 1-5, so ratings of 0 are treated as missing ratings instead. This avoids our ratings being biased downwards when performing certain operations.
3. We then added a new column, `avg_rating`, which consists of the average rating for the recipe in the corresponding row.

4. We split the `nutrition` column into multiple seperate columns.
   - The original `nutrition` column contained what looked like a list (of several nutritional values), but was actually a string/object. We created a function that turned the `nutrition` column into a list of floats, then turned each individual nutrition fact into its own column. This allows us to do a lot more exploratory analysis with each nutrition fact.
5. We added a `review_length` column. 
   - This column just contains the length of the review text. We planned on doing some analysis with the review column, so it made logical sense to add this column.
  
#### Resulting Dataframe
After these cleaning steps, we are left with a dataframe which has 234,429 rows and 26 columns. Here are the first 5 rows of our cleaned dataframe with a few columns:

| name                                 |     id |   minutes |   rating |   calories(#) |   review_length |
|:-------------------------------------|-------:|----------:|---------:|--------------:|----------------:|
| 1 brownies in the world    best ever | 333281 |        40 |        4 |         138.4 |             254 |
| 1 in canada chocolate chip cookies   | 453467 |        45 |        5 |         595.1 |             336 |
| 412 broccoli casserole               | 306168 |        40 |        5 |         194.8 |             469 |
| 412 broccoli casserole               | 306168 |        40 |        5 |         194.8 |             162 |
| 412 broccoli casserole               | 306168 |        40 |        5 |         194.8 |             188 |

More specifically, the 26 columns are:
`['name',
 'id',
 'minutes',
 'contributor_id',
 'submitted',
 'tags',
 'nutrition',
 'n_steps',
 'steps',
 'description',
 'ingredients',
 'n_ingredients',
 'user_id',
 'recipe_id',
 'date',
 'rating',
 'review',
 'avg_rating',
 'calories(#)',
 'total fat(pdv)',
 'sugar(pdv)',
 'sodium(pdv)',
 'protein(pdv)',
 'saturated fat(pdv)',
 'carbohydrates(pdv)',
 'review_length']`

### Univarite Analysis
For this part of our data analysis, we wanted to look more into the actual review texts generated by users. More specifically, we wanted to look at the length of reviews and its distribution in this dataframe.

<iframe src="assets/review_len_dist.html" 
   width="800"
   height="400px"
   frameborder="0"
></iframe>
We can see that our data is skewed to the right with some reviews having drastically longer lengths than the majority of our data. To make the visualization of this graph better, we only inluded 99% of the data since there were some extremely large outliers. We also plotted the median and mean character lengths, which were 249 and 289 respectively-another sign that our data is right-skewed.

### Bivariate Analysis
For this section of our data analysis, we wanted to look at the distribution and statistics of review text length, in regards to the rating categories. We have seen some statistics on review length in our dataset overall, but how does it look when broken down into each rating category?
<iframe
  src="assets/avg_review_length_by_rating.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
<iframe
  src="assets/review_length_by_rating_violin.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
