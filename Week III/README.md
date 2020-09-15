
# The Third Part - Algorithms

### List of Contents
[Linear Regression](#linear-regression)
[Logistic Regression](#logistic-regression)
[Estimating Accuracy](#estimating-accuracy)
[Random Forests Regression](#random-forests-regression)
[K-means clustering](#k-means-clustering)

## Linear Regression

To begin with, we'll look back at something we already discussed - linear regression., the simplest of machine learning algorithms.

Suppose we have a variable Y (say, the sales received by a company after advertising) that we want to predict using a variable X (the money the company spent on TV advertisements).  We can model the relationship between the two as:

$$Sales ≈ β_0 + β_1 \times TV$$  or $$Y ≈ β_0 + β_1X$$

β<sub>0</sub> and β<sub>1</sub> are the *intercept* and the *slope* respectively, and combined are the *co-efficients* or *parameters* of this model.  We use the training data to create estimates of these values (hence why we don't use the normal '=' sign), and then we can use them to predict company sales for potential spending on advertisements.  You can see how that might be useful.

This process of finding the strength of the relationship between variables is what we call *regression*.

#### Estimating the co-efficients

Now, how do we use the training data to find the values for these unknowns?  Let's say we have an *n* number of data entries of the form (x<sub>i</sub>, y<sub>i</sub>).  Now if we plot these values out on a graph, we'll want the slope and intercept of the line that's closest to these points.  There are many ways to calculate this closeness, but the most common one in this scenario is minimizing the *least squares*.

![Data Distribution graph](https://imgur.com/xKmGLWI.png)

Let ŷ<sub>i</sub> be the predicted value of Y for a data point x<sub>i</sub> among these data points (the real value being y<sub>i</sub>).  The error e<sub>i</sub> can obviously be seen as 
$$e_i = y_i - ŷ_i$$ This value is what we call the *i*th **residual** - literally, the residue left in the predicted value relative to the actual value.

Now you might think the obvious solution is to minimize these residuals.  But how would you do that?  It's not like we can generate every possible line to see which gives us the lowest residuals.  No, the solution lies in mathematics.

Imagine a random curve on a graph.  We want to find the lowest y co-ordinate that curve goes to (just like how we want to find the lowest combined value for the residuals).  How would we do that?  Well, if we take the derivative of the graph at any point, you'll get its slope (In *y = mx + c*, the derivative with respect to x would be *dy/dx = m*, where *m* is the slope), or in more intuitive terms, how the y co-ordinate is about to change.  

But if you take the *double* derivative, you get how the *slope* is about to change.  And at the lowest point on the graph, the slope briefly becomes 0 (at the very lowest point, when y doesn't change at all - imagine the top of an egg, where there's a very small area of flatness).  Before the curve reaches that lowest point, the slope is negative (it is going down, after all - the y co-ordinate is decreasing), and after that point, the slope is positive (the y co-ordinate will increase after the lowest point), so the double derivative (i.e, how the slope is changing) is positive.

So if we want to easily (from a mathematical sense, a couple of equations are easier than looking at every possible curve) calculate the lowest possible residuals, we need some way to take a double derivative of the residuals.  That's why we use the *residual sum of squares* (RSS), which is calculated as:
$$RSS = e_1^2 + e_2^2 + ... + e_n^2$$ Now, we can simply try to find the minimum value of this RSS.  If we do all the complicated mathematics of taking the derivative as 0 and double derivative as positive, we find that we get easily computable equations for what β<sub>0</sub> and β<sub>1</sub> should be to get the smallest RSS (or in other words, the lowest errors).  These equations don't give you any insight like the others (and none of these equations are stuff you'll need to memorize completely), so if you're averse to math that looks complicated, just read ahead normally.  If you're really curious about the equations, click below to reveal it.

<details>
<summary>The equations for β<sub>0</sub> and β<sub>1</sub></summary>
$$β_1 = \dfrac{\sum_{i = 1}^{n}(x_i - x̄)(y_i - ȳ)}{\sum_{i = 1}^{n}(x_i - x̄)^2}$$ $$β_0 = ȳ - β_1x̄$$
where ȳ and x̄ are the mean values for x and y:$$ȳ = \dfrac{1}{n}\sum_{i = 1}^{n}y_i, \;\;\; x̄ = \dfrac{1}{n}\sum_{i = 1}^{n}x_i$$
</details>

This idea, as it happens, is how we optimize most AI models - by finding mathematically, the lowest point on some inaccuracy measure graph.