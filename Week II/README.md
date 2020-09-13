# Part II - Actually Using It

#### List of Contents
[A Basic Machine Learning Algorithm](#a-basic-machine-learning-algorithm)

[A Machine Learning Program](#machine-learning-program)

[More about Deep Learning](#more-about-deep-learning)

[A Deep Learning Program](#deep-learning-program)


## A Basic Machine Learning Algorithm

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

## A Machine Learning Program

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


## More about Deep Learning

So we've talked about representing images as matrices of pixels.  If you've worked with images before, you'll know that colour images are actually three matrices stacked on top of each other - one for red, one for blue, and one for green - that combine together to form the image.  You could also call it a matrix with three dimensions.  We call this sort of structure a **tensor** - basically, a matrix that can have any number of dimensions.  From now on, we'll be calling all data representations of images as tensors.

Look at the graphic below.  There's a black-and-white image of the number 2.  The tensor below it is a rough approximation of what the image would be like in pure data.  Of course, the actual tensor would be bigger as there are more pixels in the image, but it would look somewhat like that.

![Image to Data](https://imgur.com/nvBezXQ.png)

We then apply convolutions to this tensor (In practice, we don't pick the numbers for the convolutions manually - that's the model's job) to get a new, smaller tensor.  There are many types of convolutions we can apply to an image - for example, we can get a new tensor identifying the horizontal lines in the image, another identifying the vertical lines, and so on, for any convolutions we may have.  We can stack these tensors together to create a new 3-D tensor.

To understood what that would look like, here's a graphic:

![Dimensions of Tensor](https://imgur.com/WW2Ryde.png)

Now, we go further and apply more convolutions to this 3-D tensor, to get another tensor with even more specific details.  Just as we use the patterns in the darkness to find lines in the first tensor, we use patterns in those lines, to find new shapes, and then keep going (which is how we can go from finding concentric circles, to wheels).

![How a Deep Neural Network Sees](https://imgur.com/vJAsvJO.png)

So the prime difference between machine learning and deep learning is that in deep learning, we don't have to bother with telling the model what features to look at - it finds them on its own.

One of the most common applications of this kind of neural network is object detection.  Deep Learning - and indeed, all of Machine Learning - work on probabilities.  The model will look at an image, and tell you what probability it estimates to it containing a certain object.  This becomes more difficult when the same object looks entirely different when viewed from different angles (a person from behind looks different than from the front).  This is why training a model requires a large number of images, to familiarize it with all the forms the object can appear in.

Basic models would struggle to achieve a high degree of accuracy, and when it comes to more similar-looking objects such as various breeds of dogs, it's even more difficult.  There are competitions where people submit their models to compete for accuracy.  These models are pre-trained, which means that we could use them without needing to find and label (very important, as the label is what tells the model what an image contains during training) images to train our own model, we can use these powerful existing ones.

But these models are trained to identify a certain kind of object.  How would a model that's used to classify various breeds of dogs, help us with identifying cats?  Well, as we've seen, the first few layers of a deep learning network, would usually find the same things (lines, shapes, etc).  So what we can do then, is remove the final layer of the model - the last set of convolutions - that does the actual classification, and train the rest of the model on our own set of images.  Because most of the model is already trained, we can achieve very good results with very little data.  This is called **transfer learning**, and it's what you'll be trying out after this. 

For now, we'll run through the code for simply using a pre-trained model to classify breeds of dogs.  We're using the ResNet-50 model, which has 50 layers (kinda obvious, given the name, isn't it?)

## A Deep Learning Program

The os library in Python is used to handle some things that the operating system has control over, such as file and folder names.  Here, we import the join function to create a list of paths to images.

The images are present in the `train/` folder, and their filenames are those long random strings in the third line of code.  After this code is executes, `img_paths` contains four entries - the paths to those four images.

```python
from os.path import join

image_dir = '../input/dog-breed-identification/train/'
img_paths = [join(image_dir, filename) for filename in 
                           ['0c8fe33bd89646b678f6b2891df8a1c6.jpg',
                            '0c3b282ecbed1ca9eb17de4cb1b6e326.jpg',
                            '04fb4d719e9fe2b6ffe32d9ae7be8a22.jpg',
                            '0e79be614f12deb4f7cae18614b7391b.jpg']]
```

Numpy is a library that gives us support in handling large, multi-dimensional arrays (and provides quite a few complex mathematical functions we can do on them, but we don't need that now).

TensorFlow, much like sci-kit learn, is a library that's used for many machine learning applications - including deep learning, which it uses Keras, a library built specifically for neural networks. 

The `load_img` function loads the image from the file path given.  The `img_to_array` function creates the 3-D tensor for each image (we're dealing with colour images), and stores them in an array, creating a 4-D tensor (the extra dimension being the number of images, here four).  The `preprocess_input` function is used to normalize the pixel values of image to be between -1 and 1, for easy handling (and consistency, because that's how the model was first built).

As the model was trained with 224\*224 resolution images, we'll be using the same resolution here.

Now we write a function `read_and_prep_images` to load the images, convert them into 3-D tensors, and normalize their values.

If all this sounds long and/or complicated, don't worry.  They'll become more familiar as you run the commands on your own and get used to them.

```python
import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)
```

As we create the model for us to use, we give an argument specifying the file path to the values in the convolutional filters (as the model is pre-trained, they already exist).

Now we simply prepare our images with the function we saw earlier, and run it through the model.

```python
from tensorflow.python.keras.applications import ResNet50

my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)
```

Now, for each image, the model provides the probabilities for each different type of breed it could be.  As there are quite a few, let's just look at what it thinks are the three likeliest options for each image.

The `decode_predictions` function provided by Keras to extract the highest probabilities for each image.  The `class_list_path` links to the classes the ResNet model is trained for (or in other words, the labels given to the images it was trained on).

To view the image and what the model thinks it contains, we import the functions `Image` and `display` to, well, display the images, obviously.

```python
from learntools.deep_learning.decode_predictions import decode_predictions
from IPython.display import Image, display

most_likely_labels = decode_predictions(preds, top=3, class_list_path='../input/resnet50/imagenet_class_index.json')

for i, img_path in enumerate(img_paths):
    display(Image(img_path))
    print(most_likely_labels[i])
```
Output:

![Picture of Dog](https://imgur.com/RZ3sy85.png)
![Prediction 1](https://imgur.com/Tfrss0q.png)
![Another Picture of Dog](https://imgur.com/ppAKn1l.png)
![Prediction 2](https://imgur.com/ukcwSMu.png)
![Yet Another Picture of Dog](https://imgur.com/FPjrNQB.png)
![Prediction 3](https://imgur.com/jRyjtXG.png)
![Picture of Cat.  Nah, I'm kidding, it's a Dog.](https://imgur.com/l4bRc5r.png)
![Prediction 4](https://imgur.com/y1qjHUJ.png)

The probability of a picture being each breed is given by the number after the name in each set.

Now, it's time to [try something different](https://www.kaggle.comexercise-transfer-learning), with transfer learning.