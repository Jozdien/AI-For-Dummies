# The Tools of Artificial Intelligence

Intelligence is the use of information and logic to solve problems and learn new things.  How exactly you do that, is a question with many answers, which is why you might have heard of machine learning, deep learning, and possibly others if you had nothing better to do than look this stuff up.  

All of them achieve the same basic goal - making a computer learn how to solve problems - through very different methods.  Here, we'll look at what those methods really are, in the simplest possible methods, and what we can do with them.

* [Machine Learning](#machine-learning) (ML) is the process by which a computer learns and improves from experience, to make predictions or decisions, without being explicitly told how to. 

* [Deep Learning](#deep-learning) (DL) is the process by which a computer learns through representing data in the form of increasingly simplified levels.  Does that sound confusing?  Don't worry, it'll make sense when it's explained.  Probably.  DL is technically a section of ML, but it's grown so large in itself that it's worth mentioning separately.  

##  Machine Learning

Machine Learning comprises a large group of methods by which a computer can learn from data, so here, we'll look at what those methods are really doing.  

At its core, ML is about using algorithms to analyse and predict patterns in data.  If a computer is given a set of data about the heights and weights of different people, and we want to use that to predict someone's height using just their weight, then we train the computer on the data we do have, so that it can have a decent understanding of the correlation between height and weight.  How does it do that? Magic*.

*More accurately, Math.

Despite what some say, the math really is simple at the core.  You take a graph of variables, and plot a line connecting the points on it.  Look at this graph, for instance: 

{Image}

If you wanted to predict what Y would be for a new value of X (e.g: 16), how would you do it?  (Seriously, how would you do it?  I can't remember.)

So after some a few Google searches, it turns out that you draw a straight line roughly connecting all the points, and then look at what Y the new X would give on that line, and you can see it comes pretty close to the actual answer.

{Image}

That's Machine Learning.  No seriously, that's the basic idea.  You give the computer a lot of data like this, and it "understands" it by finding patterns in them. 

### So when does it get more complicated?

Well, you might have noticed that the predicted value of Y isn't **exactly** the same as the actual value.  Now, because people are picky as hell, they want more accuracy.  So we end up with patterns like:

{Image}

Now here, we fit exactly each point to a weird, zig-zaggy line.  But if you were looking closely, you'd see that here too, the predicted value of Y is slightly wrong.  This is because in a real-world scenario, there are more variables involved than the ones we're looking at, so individual data points may be affected in different ways, and trying to get a perfect fit will lead to errors because of the over-reliance on those individual points.  Think of it as a group of people working together toward some goal.  When each of them has less individual power, their common goal becomes more prominent, while if everyone had more power, they'd try to push things toward their own personal goals.

Thus, simply connecting the data with a straight line can get you wrong answers because you're not accurate enough, and connecting it with too perfect a fit can get you wrong answers because you're trying too hard to be accurate.  One of the real problems in Machine Learning comes in trying to hit the perfect middle.  For example, look at the following graph:

{Graph}

As you can see, here we have a curvy graph that's more flexible than a straight line, but doesn't try to be a perfect fit.  Thus, we have a predicted value of Y that's closer to the actual value than we did before.

### In the Real World

Of course, in real-world situations, there are way more than 2 variables.  The complexity of ML, and the reason it's still growing as a field, is because computer models are growing ever larger.  

A good example would be something like the recently released GPT-3, a text processing engine which uses *175 billion* parameters (they're called parameters because the method used is closer to Deep Learning, which we'll see later).  GPT-2, its predecessor, is slightly less massive with only 1.5 billion parameters, but still does some amazing, weird, brilliant stuff.  Check out [AI Dungeon](https://play.aidungeon.io/) if you want to read/write some of the craziest stories you'll ever find.  If you're familiar with Reddit, try this [subreddit filled entirely with GPT-2 bots built on different subs](https://www.reddit.com/r/SubSimulatorGPT2/).

## Deep Learning

How would you teach a baby what a "dog" is?  I doubt you'd get very far trying to explain to them all the features of a dog, and expecting them to understand.  No, you'd point to a dog, and say "That's a dog".  Now the baby associates the visual input it gets in that moment with the word "dog".  As they grow up and learn that more types of breeds are also called "dogs", their idea of what a dog is becomes more general - they learn to pick up on the unique traits that make up a dog purely from the experience of associating certain things as dogs.

That's the idea behind deep learning.  When machine learning first came to the mainstream, it was amazing.  It was powerful, and it got great results.  But people saw that it didn't quite work like human minds do.  Hence, the development of this field, where you teach a computer to associate certain characteristics with a concept through simplifying visual data into simpler forms, like you'd simplify the sight of an animal into characteristics such as "four legs", "tail", "dull claws", and simplify those further into "dog".