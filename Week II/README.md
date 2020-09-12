# Part II - Actually Using It

#### List of Contents
[A Very Basic Machine Learning Algorithm](#a-very-basic-machine-learning-algorithm)

[Machine Learning Program](#machine-learning-program)


## A Very Basic Machine Learning Algorithm

Let's look at a very basic machine learning algorithm: Decision trees.

Imagine you finish this workshop, and go on to become a data scientist.  A friend who works in real estate asks you for your help in estimating the price a house would sell for.  You ask him how he normally does it, and he calls it intuition.  You ask a little more, and he tells you that there are certain obvious factors that factor into a house's value - more rooms means higher value, etc.  

That's kind of reasoning is what we use in decision trees.

![Depth 1 Tree](https://i.imgur.com/7tsb5b1.png)

This, for example, is a simple decision tree that a computer can infer from data (the actual values will of course depend on the dataset itself).  Creating it could be as simple as taking the average values of houses with two or less bedrooms, and houses with more than two bedrooms.

The computer uses the data to decide how to break the houses into two groups, and also to determine the predicted price in each group.  This process is called **training**, and the data used for this is **training data**.

But here's a problem: remember that overfitting we mentioned last time? (If you don't remember and are too lazy to go back and see, it's when a model trains too well on the data, and becomes very accurate for points in that data, but less accurate for new data)  Well, the problem persists.  The solution here, is the standard practice of dividing the dataset into two sections: the **training data**, and the **testing data**.  Now, we train the model only on the first section of data, and test it on the second, which it hasn't seen before.  If it performs poorly, we retrain it (again on only the first section) with new parameters to try and minimize its inaccuracies.

But we're getting ahead of ourselves.  For now, we'll deal with only training data.  Let's go back to looking at that tree.

Of course, this kind of model wouldn't really help us all that much.  There are many more factors at play, and only considering this one gives us very little information.  So let's try branching out a little.

![Depth 2 Tree](http://i.imgur.com/R3ywQsR.png)

Now this tree is a little more helpful.

As we have an increasing amount of data and features to examine, we can continually create more and more accurate predictions about the house's value.  More advanced models combine multiple random trees to create a better approximation of the data (groups of trees are what we call forests, and this method is called Random Forests Regression, which we'll look at next time).

## Machine Learning Program

At the end of this, there's a link to a site where you can actually run a machine learning program by yourself, and see what you get.  What follows this is just a guide to help you understand what you'll be running.

To begin with, we import two packages to help us out.

The first is the pandas library, helpful in easy handling of data. The second is the sklearn library, a machine learning library that gives us the algorithms we need. Here, we'll be using the decision tree regressor, a basic algorithm that functions on the same principle of trees that we just went through (yeah that's right, you can't just skip to this section and skip the previous stuff).
```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor 
```
Now what we need is a dataset.  Luckily, there are many freely available datasets online (because who has the time to make one of these of our own), and we'll be using one of them here - consisting of data about Melbourne real estate.  They're in CSV format, and luckily, the pandas package we imported allows us to easily read it from the file path using the `pd.read_csv()` function.  Now let's look at the columns we have in the data (some of which are what we'll be using as features in training the model later on).

```python
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.columns
```
Output:
```
Index(['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
       'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount'],
      dtype='object')
```

Now let's assign the target value (here, that would be the price of the house) to a variable y.

```python
y = melbourne_data.Price
```

We need to select a few features to work on.  Selecting too many features - especially useless ones - can result in overfitting, so you have to be careful to pick only as many as you need (of course, too few and the model won't be accurate at all, so, you know, be careful).

This data is stored in a variable x.

```python
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

x = melbourne_data[melbourne_features]
```

As a data scientist, one of the biggest tasks is visually checking your data to find any immediate problems that'll fuck everything up.  `.describe()` is a function to view a summary of the feature data.

```python
x.describe()
```
Output:

|       |    Rooms    |   Bathroom  |   Landsize  |   Latitude  |  Longitude  |
| -----:| -----------:| -----------:| -----------:| -----------:| -----------:|
| count | 6196.000000 | 6196.000000 | 6196.000000 | 6196.000000 | 6196.000000 |
| mean  |  2.931407   |   1.576340  |  471.006940 | -37.807904	|  144.990201 |
| std   |  0.971079   |   0.711362  |  897.449881 |   0.075850  |   0.099165  |
| min   |  1.000000   |   1.000000  |   0.000000  |  -38.164920 |  144.542370 |
| 25%   |  2.000000   |   1.000000  |  152.000000 |  -37.855438 |  144.926198 |
| 50%   |  3.000000   |   1.000000  |  373.000000 |  -37.802250 |  144.995800 |
| 75%   |  4.000000   |   2.000000  |  628.000000 |  -37.758200 |  145.052700 |
| max   |  8.000000   |   8.000000  | 37000.00000 |  -37.457090 |  145.526350 |

The max landsize is pretty big compared to the rest, but seeing as that's the only problem, we can let it slide.

`.head()` is another function that's used to visualize the data, and this one returns the first five rows of the data, to see if there are any immediate anomalies that pop up.

```python
x.head()
```
Output:
|   | Rooms | Bathroom | Landsize |  Latitude  | Longitude |
| -:| -----:| --------:| --------:| ----------:| ---------:|
| 1 |   2   |    1     |  156.0   |  -37.8079  | 144.9934  |
| 2 |   3   |    2     |  134.0   |  -37.8093  | 144.9944  |
| 4 |   4   |    1     |  120.0   |  -37.8072  | 144.9941  |
| 6 |   3   |    2     |  245.0   |  -37.8024  | 144.9991  |
| 7 |   2   |    1     |  256.0   |  -37.8060  | 144.9954  |

Looks like we're clear here.  Now let's actually use the model on this data.

`DecisionTreeRegressor()` is a function available in the sklearn library.  There are quite a few parameters we can change by giving values between the brackets if we wanted to, but for now, we'll just use the `random_state` parameter.  Most algorithms have a certain degree of randomness to it.  Specifying a number like we're doing here, ensures that we'll receive the same results in each run.  This is considered good practice, because unlike what we mentioned earlier about combining multiple random trees, this is just one tree with more random values each time.

The `.fit()` function is used to train the model on the data we've prepared for it.

```python
melbourne_model = DecisionTreeRegressor(random_state=1)

melbourne_model.fit(x, y)
```

If that felt underwhelming, it's because it was.  A good portion of machine learning is simply preparing the data you have to be processed.  Once you're done with that, the code is actually a relatively simple matter.

Now finally, after our model has been trained, we can check to see if it's actually worked.  We'll try it with the first five houses in the dataset, and see what values our algorithm estimate they would have.

```python
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
```
Output:
```
Making predictions for the following 5 houses:
   Rooms  Bathroom  Landsize  Lattitude  Longtitude
1      2       1.0     156.0   -37.8079    144.9934
2      3       2.0     134.0   -37.8093    144.9944
4      4       1.0     120.0   -37.8072    144.9941
6      3       2.0     245.0   -37.8024    144.9993
7      2       1.0     256.0   -37.8060    144.9954
The predictions are
[1035000. 1465000. 1600000. 1876000. 1636000.]
```
Now, [try it for yourself](https://colab.research.google.com/drive/1owQJW9awNasTYXw68RO5iHyw23LT-5uB?usp=sharing).