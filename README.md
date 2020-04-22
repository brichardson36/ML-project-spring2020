## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/brichardson36/ML-project-spring2020/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List



**Bold** and _Italic_ and `Code` text


[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

# Introdction
The evolutionary arms race of stock trading continually drives the development of new machine learning methods. What we propose to do, is to take an artificial neural network known as an LSTM (Long Short-Term Memory) and use it in lieu of traditional trading models.
# Methods
## Stock Selection
The data chosen were the intraday Trade and Quote prices and volumes of MA (MasterCard) over the days of May 2016 (source: WRDS).
## Preprocessing
### Discretization
To allow for consistent forecasting, and to eliminate the impact of tiny discrepancies of time as a pure input, the data was binned into discrete buckets.
### Bucket Description
The buckets spanned from 9:30 AM to 4:00 PM, every business day. Each bucket was 1 minute long, and contained the opening, closing, high, and low price, as well as volume traded, for the bucket. The width of the buckets was minimized while ensuring every bucket had at least one data point.
### Feature Engineering
This process created a slew of features for models. The data need to be processed again, however. This stems from Stationarity. Stationary data has a constant mean and standard deviation over a given time period [sources needed] (reference pure_price_kde figures). Forecasting non stationary data is unreliable [sources needed]. To create Stationary data, the difference between a bucket’s parameters and the previous bucket’s parameters was calculated. This difference for a given bucket was calculated for all 5 previous buckets to create more features. The difference data was stationary (reference diff_in_price_kde figures) and thus suitable as a set of parameters for forecasting models.
### Normalizing
Data was normalized by subtracting the average of feature_n from each feature_n, and dividing the subsequent feature_n by the standard deviation of feature_n. This reduces scaling errors from features that have different units and scales. In this case, the scaling error between price and volume is reduced.
## Metrics
Models are successful if they return a higher percent profit over the time frame than comparable methods. To quantify this success, we chose two metrics: average profit per day and average ratio of correct purchases to total purchases per day.
# Models & Results
## Ridge Regression
### Reasoning
We assume that a price at time = n is related to a price at time = n+1, and that the differences in prices likewise reflect a dependent relationship. The parameters will suffer from multicollinearity. To tackle multicollinearity, ridge regression was proposed as a model. Ridge regression is a regression model built to account for multicollinearity. In ridge regression, a constant is added to all diagonal indices of the correlation matrix to improve variance.
### Model implementation
In the model used here, the constant to add was found using scikit-learn’s auto setting. All differences in open, close, high, low prices as well as volumes were passed as parameters, in addition to prices. The model was trained to predict the open price at one and two timesteps of 1 min in the future from the current open price.
### Bot Implementation
The trading bot took in parameters as listed above. When the model predicted a price increase from the next open price to the subsequent open price, it would purchase one stock. When the model predicted a decrease in a similar fashion, it would sell all the stock. Selling all the stock increases the chance to be fully exited from one’s market position by the end of the day, and encourages the bot not to be forced to sell all the stock at a closing price.
### Results
avg profit per day, median and mean
$1.47
$7.01
avg stocks traded per day, median and mean
291
259
avg correct predictions per day, median and mean
95
124
avg incorrect predictions per day, median and mean
161
134
avg ratio of correct to total predictions per day, median and mean
47 %
48 %
The ridge regression model and bot produced $7.01 dollars every day on average, with no cap on how much money is invested at any given time. From the graphs we can see most of the change in profit came from two days. This implies that the model is not consistent. The model also correctly predicted prices only 48% of the time. Let’s see if we can do better - both more consistent, and more correct predictions.

