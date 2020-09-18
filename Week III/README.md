# The Third Part - Algorithms

### List of Contents
[Linear Regression](#linear-regression)
[Logistic Regression](#logistic-regression)
[Random Forests Regression](#random-forests-regression)
[Choosing Models](#choosing-models)
[Estimating Accuracy](#estimating-accuracy)

## Linear Regression

To begin with, we'll look back at something we already discussed - linear regression., the simplest of machine learning algorithms.

Suppose we have a variable Y (say, the sales received by a company after advertising) that we want to predict using a variable X (the money the company spent on TV advertisements).  We can model the relationship between the two as:


![Equation 1](https://imgur.com/tyAD01S.png) 

or

![Equation 2](https://imgur.com/Yvsqv4q.png)

β<sub>0</sub> and β<sub>1</sub> are the *intercept* and the *slope* respectively, and combined are the *co-efficients* or **parameters** of this model.  We use the training data to create estimates of these values (hence why we don't use the normal '=' sign), and then we can use them to predict company sales for potential spending on advertisements.  You can see how that might be useful.

This process of finding the strength of the relationship between variables is what we call **regression**.

#### Estimating the co-efficients

Now, how do we use the training data to find the values for these unknowns?  Let's say we have an *n* number of data entries of the form (x<sub>i</sub>, y<sub>i</sub>).  Now if we plot these values out on a graph, we'll want the slope and intercept of the line that's closest to these points.  There are many ways to calculate this closeness, but the most common one in this scenario is minimizing the *least squares*.

![Data Distribution graph](https://imgur.com/xKmGLWI.png)

Let ŷ<sub>i</sub> be the predicted value of Y for a data point x<sub>i</sub> among these data points (the real value being y<sub>i</sub>).  The error e<sub>i</sub> can obviously be seen as 

![Equation 3](https://imgur.com/90WlYFg.png)

This value is what we call the *i*th **residual** - literally, the residue left in the predicted value relative to the actual value.

Now you might think the obvious solution is to minimize these residuals.  But how would you do that?  It's not like we can generate every possible line to see which gives us the lowest residuals.  No, the solution lies in mathematics.

Imagine a random curve on a graph.  We want to find the lowest y co-ordinate that curve goes to (just like how we want to find the lowest combined value for the residuals).  How would we do that?  Well, if we take the derivative of the graph at any point, you'll get its slope (In *y = mx + c*, the derivative with respect to x would be *dy/dx = m*, where *m* is the slope), or in more intuitive terms, how the y co-ordinate is about to change.  

But if you take the *double* derivative, you get how the *slope* is about to change.  And at the lowest point on the graph, the slope briefly becomes 0 (at the very lowest point, when y doesn't change at all - imagine the top of an egg, where there's a very small area of flatness).  Before the curve reaches that lowest point, the slope is negative (it is going down, after all - the y co-ordinate is decreasing), and after that point, the slope is positive (the y co-ordinate will increase after the lowest point), so the double derivative (i.e, how the slope is changing) is positive.

So if we want to easily (from a mathematical sense, a couple of equations are easier than looking at every possible curve) calculate the lowest possible residuals, we need some way to take a double derivative of the residuals.  That's why we use the **residual sum of squares** (RSS), which is calculated as:

![Equation 4](https://imgur.com/qbksy6a.png)

Now, we can simply try to find the minimum value of this RSS.  If we do all the complicated mathematics of taking the derivative as 0 and double derivative as positive, we find that we get easily computable equations for what β<sub>0</sub> and β<sub>1</sub> should be to get the smallest RSS (or in other words, the lowest errors).  These equations don't give you any insight like the others (and none of these equations are stuff you'll need to memorize completely), so if you're averse to math that looks complicated, just read ahead normally.  If you're really curious about the equations, click below to reveal it.

<details>
<summary>The equations for β<sub>0</sub> and β<sub>1</sub></summary>

![Equation 5](https://imgur.com/5IhQ9ky.png)

where ȳ and x̄ are the mean values for x and y:

![Equation 6](https://imgur.com/k3L787k.png)

And as a bonus, this is what the graph for that would look like.  The red dot is the lowest point in that graph and thus, the lowest value of RSS, and β<sub>0</sub> and β<sub>1</sub> are simply the x and y co-ordinates of that point.

![RSS Graph](https://imgur.com/YHYzPTM.png)

</details>

This method, as it happens, is how optimization for most AI models are performed - mathematically finding the lowest point on a graph measuring inaccuracy.

If you want to try out the code for this method yourself (or just see what the program looks like), [here's a colab link](https://colab.research.google.com/drive/1fH9kVh-L0gXHEjOoNYdMu4bCoU1Hk3xF?usp=sharing) (disclaimer: I'm not the author, so it's not written explicitly with the flow).

### Multiple Linear Regression

Maybe you're a large company.  You have the budget to spend on more than TV advertisements.  Maybe you advertise in the newspaper and on billboards as well.  How would you then find out the optimal amount to spend? 

![Equation 7](https://imgur.com/SkiYxm2.png)

Here, β<sub>i</sub> stands for the relationship between X<sub>i</sub> and Y, independent of all other variables.  In other words, β<sub>i</sub> is the change in Y for one unit of change in X<sub>i</sub>, everything else remaining the same.

Now I'll tell you something weird that happens, and you try and see if you can predict why it happens.

Remember when we plotted a graph with a lot of data points for the simple linear regression between sales and money spent on TV advertisements?  Well, the graph looks good, doesn't it?  Clearly shows that if you spend money on TV, you make money.

But now when we try it with this model, we find that somehow, β<sub>1</sub> is close to 0; or more simply, sales does not change when we spend more on TV advertising.  Can you guess why?

<details>
<summary>The answer</summary>

Well, it turns out that whenever your marketing team spends more money on TV, they also spend more on newspapers and billboards.  In the simple linear regression, you aren't just calculating how much sales will increase when TV advertising increases - you're calculating how much sales will increase when every situation where TV advertising *could* increase occurs.  Now, in the multiple linear regression, where you take the change in the TV spending while newspaper and billboard budgets remain the same, you see that it's not very relevant at all.
</details>

The fundamental principle at work here, is that **correlation is not the same as causation**.  It can be though, and your intuition can't always be trusted, so your best bet is usually to test and experiment with other variables - the truth is often counter-intuitive.  

But fortunately, when you run a regression of car accidents and KFC sales, you would find a positive relationship, similar to that between TV and sales.  The real reason is simply that on days when there are more people outside, there are more road accidents, and KFC has more customers.  The fortunate aspect is that sometimes, in rare instances, its obvious enough that no one's tried to ban KFC to reduce accidents.  Well, no one yet.

## Logistic Regression

Let's say you're a doctor.  A patient is wheeled into the emergency room after collapsing, and you need to find out whether they've had a stroke, an epileptic seizure, or a drug overdose.  There are a lot of factors that could go into predicting it (many X<sub>i</sub>), but that's not the focus here.  The equation would look something like:

![Equation 9](https://imgur.com/0AO7Are.png)

You could figure that if the predicted answer is the one that Y is calculated closest to - if it was 1.4, it would be a stroke, for example.  But the problem is that if you let Y be, say:

![Equation 10](https://imgur.com/cQg8ftO.png)

Then the values for the parameters would be entirely different.  This becomes a problem as we get different predicted values on test data from pointless things like this.

That isn't a problem when you only have two variables.  Then it would look like:

![Equation 8](https://imgur.com/V1AAjWl.png)

So we could say that it's a stroke if Y is less than 0.5, and a drug overdose if it's greater than.  In other words, we could treat it as the probability of a drug overdose.

But what if Y goes above 1?  Or below 0?  The *factors* (X<sub>i</sub>) aren't bound by limits in real life, so in some extreme scenarios, it's possible that the equation for Y in terms of the X<sub>i</sub>s gives us values that probabilities wouldn't have, kinda ruining the analogy.  It provides a crude prediction, but crude isn't what we're going for.

Turns out there are a lot of functions that can give outputs between 0 and 1 for all inputs.  In logistic regression, we rather predictably use the logistic function to model *p(X)*, the probability that Y is one of the two possibilities given a certain value of X (we'll look at the scenario where there's only one factor first).

![Equation 9](https://imgur.com/Y9j8ogz.png)

There's no reason we chose this equation other than to get that output range, so it's not necessary to think about the mathematics behind this.  

To compare, consider the scenario where you want the probability that a person would default on their loan, with the parameter being the balance in their account.  It would look something like the following graph (the orange marks are the balances for which someone defaulted), where the left is linear regression, and the right logistic regression.

![Linear-Logistic Graphs](https://imgur.com/xTbei72.png)

The right one does look more accurate to the data, doesn't it?

If you do a couple interesting things to that logistic function equation, you get this:

![Equation 10](https://imgur.com/HryhE8b.png)

That value on the left is called the *odds*, and it's what's used in traditional betting.  Like when people say the odds of something are 9 to 1, they mean the probability is 0.9, which gives a value of 9 on the left hand side.  Just an interesting tidbit, you can move on now.

By taking the logarithm on both sides (because why not, honestly), we get:

![Equation 11](https://imgur.com/AYtATIu.png)

Now, the value on the left is what we call the log-odds.  Surprisingly, you don't see many traditional gamblers use this term, for (math) whatever (it's the math) reason (It's because of math).

Now, the terms on the right are the same old linear equation from before.  And with two parameters, just like before.  To find that, there's something called a **likelihood function** which is basically just finding the values for the parameters for which the value of *p(X)* (not the left hand side of the last equation, but the probability of default) is closest to 1 for actual defaulters in the training data and closest to 0 for non-defaulters.  Simple, isn't it?  The math isn't as much, and if you're still interested in that at this point, you can find it with a fairly simple Google search (I think for the rest, there are more than enough equations here).

If there are more factors, we can simply substitute them into the equation, and we get:

![Equation 12](https://imgur.com/TcIRmEB.png)

Lastly, what if we have more than two options?  What if instead of dealing with a scenario where a person can default or not, we're looking at the same old scenario of a collapsed patient in a hospital?  The models we looked at just now can be applied to more classes, but they're not used that often in that form.  That's partly due to the existence of better methods, such as *discriminant analysis*.  

### Linear Discriminant Analysis

Including the formal theory for that here as well would make this longer than it already is, so I'll just explain it intuitively.

First, why is this better (apart from the ability to handle multiple classes)?  

Well, when the classes are very separated, logistic regression isn't as strong, as the parameters become very unstable - like if in the defaulters scenario, there was a stronger divide between the defaulters and non-defaulters - the graph would at some point take a sudden, sharp turn upward, instead of a slow gradient at the point where the orange ticks faded from the default line.  The parameters go to negative infinity and infinity respectively, to fit that curve, which you might realize is not a realistic relationship between the variable and the output.

If the size of the training data is small, this method performs better.

Now, what does this model do?  Well, it first models the distribution of X in each output class.  If we consider the patient scenario again, it's like finding the probability distribution of his age (let that be a factor) if he had a stroke (which would be a graph that probably leans toward the older side) as well as for the other possibilities, repeated for all the factors.  Basically, we find all *P(X)* s given *Y* = a specific class *k*.  (It can be hard to picture this.  Take a second here.)

Now we use [Bayes' theorem](https://arbital.com/p/bayes_rule/?l=1zq) to flip these to get *P(Y = k)*, the probability distribution for *Y* being a specific class *k*, for all values of *X*.  Now we have a distribution that tells us what the probabilities are for the cause of the patient's condition, given the values of the variables.  (If you don't remember what Bayes' theorem is - or even if you do, I highly recommend checking out that link, which is a site that explains mathematical concepts in a very intuitive, easy way.)

## Random Forests Regression

How do trees work?  When we last saw one (all the way back in Part II), it reduced the entire extent of the output into two neat values, depending on one variable.

Roughly speaking, building any tree involves two mathematical steps. :

* Dividing the set of possible values for the parameters into *n* different regions (If there's only variable, it'll be *n* lines, if there are two, it'll be planes, if there are three, it'll be boxes, and so on).
* For any new observation, we simply find the region it falls into, and assign it an output value equal to the average of the training data in that region.

Now, we will obviously want to construct the regions in a way that minimizes the RSS as much as we can.  But how do we do that?  We can't exactly look at every possible set of regions to see which is the best.  Well, we can, but it'd be a lot of work, and wouldn't it be nicer if we could find an easier way?

There's a method known as **recursive binary splitting**.  It's *top-down*, which means we begin at the top of the tree and then keep splitting each region into two smaller ones; each split leaves behind two new branches.  It's *greedy* (yes, it's actually called that), which means that at each step, it finds the best possible split at that particular step, instead of looking ahead to see what would be better in the future.  Rather short-sighted and selfish.

So at every step, we find a variable *s* for every parameter, such that the division into the regions where *X<sub>i</sub>* > *s* and *X<sub>i</sub>* < *s* for all *X<sub>i</sub>* (*s* is obviously different for each) gives us the greatest reduction in RSS.  In other words (or rather, equations), we want the combined RSS for the two new regions to be as low as possible.  This process takes way less time than coming up with random divisions of the regions.

Of course, if we keep dividing into regions while looking for the lowest RSS, we'll end up with a very overfitted tree.  A smaller tree might have more bias, but it'll also have far less variance (Look at Part II if you don't remember what that means).  We can't even set a condition that the decrease in RSS has to be greater than a certain value at each split, because a low-quality split may be followed by a much better one.

A better strategy may be to grow a tree to a large size, and then cut it down to a more manageable size.  For this, we use a method known as **cost complexity pruning** or **weakest link pruning**.  The logic is, instead of simply trying to reduce the RSS, we try to reduce the sum of the RSS and some term that increases with the size of the tree.  Makes sense, doesn't it?  The larger the tree, the larger that term will become, so when we're cutting down the size, we just need to look for a subtree for which that sum is the smallest.  

The term we use here is *α|T|*, where *|T|* is the number of leaf or terminal nodes on the tree (the nodes that have no further branches sticking out of it), and *α* is a tuning variable we can adjust as necessary.  If we want a smaller tree, we give *α* a large value, and if we want a larger tree, we give *α* a smaller value.  If *α* was 0, then the subtree with the smallest RSS would simply be the tree itself, and we can't cut anything down, which isn't fun.  We can now find the ideal value for *α* using some method of estimating accuracy (which we'll see next), because the cutting down of the tree progresses rather predictably for increasing values of *α*, and thus getting a sequence of subtrees as a function of it is possible.

### Random Forests

What if one tree wasn't enough?  What if you wanted to go *really* overboard with it, and make a *lot* of trees?  After all, if one tree is good, averaging a lot of them can only be better.

Now how do we make these different trees?  One method, known as **bagging**, involves selecting *n* random points from the *n* different *(X,Y)* pairs, with repetition allowed so the selections aren't the same.  This decreases variance, as specific quirks are accounted for in the averaging of different types of trees (of course, some quirks exist throughout the entire training data, and there this won't help), without increasing bias as we're not retraining a tree on the same dataset.

But this comes with its own problems.  Namely, if there's one or two *very* strong predictors in the model along with a few moderately strong ones, then when we use bagging, nearly all the trees formed will use those strong predictors, and subsequently, those trees would look very similar, meaning that variance doesn't increase all that much.  This problem comes mostly because we're always selecting *n* points, because then it's more likely that they would contain that strong predictor.

So to solve this, we could simply take a subset of the predictors.  Instead of selecting *n* random points, we could select a smaller number *m*, and train trees with that number.  Because we're using a lot of trees, we don't lose power from the smaller training data per tree.

This process is called **random forests regression**.  Lots of trees, and its random, so the name makes sense.  The value for *m* is typically taken as *√n*.

## Choosing Models

If the relationship between the predictors (factors) and the output is linear, you use linear regression.  If the relationship is non-linear and more quadratic, logistic regression works well.  If it's very non-linear and complex, trees might be a better option.  

![Linear Regression vs Trees](https://imgur.com/RB9Ps5j.png)

Trees are popular for more reasons than their predictive power - they're easy to understand by looking at them, and that's always nice.  Unfortunately, they aren't quite as accurate as some of the other regression techniques, and even a small change in the training data can result in a large change in the final tree.

## Estimating Accuracy

As we already saw, training error rate is quite different from testing error rate.  The test error rate can be calculated if we have a dedicated testing set is available, but that's not always the case.  In that scenario, assessing a model using its training error rate can lead to overconfidence in its accuracy.  To finish off Part III, we'll look at some of the methods for estimating the test error rate by holding out on some of the training data.

The obvious solution is to split the training dataset into two parts, and use one for testing, a solution known as the **validation set approach**.  But the problem that arises is that the model's test accuracy will depend heavily on *which* data is in the first part, the part its trained on.  And since models are inclined to perform better with more training, this might be a situation where if we train it on only a part of the data, it'll actually *overestimate* the test error.

What if we could get more training while also keeping our accurate tests?  That's the idea behind **Leave-one-out cross-validation** (That's a lot of words, so let's just call it LOOCV).  In it, we train the model on the entire training data, excluding just one observation, which we'll use for testing.  And since the test error (the mean square error MSE = *(y<sub>i</sub> – ŷ<sub>i</sub>)<sup>2</sup>*) here will be very variable because we're using just one observation instead of averaging over an entire testing dataset, we we repeat this for every single observation.  