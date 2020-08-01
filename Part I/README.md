# The Tools of Artificial Intelligence

Intelligence is the use of information and logic to solve problems and learn new things.  How exactly you do that, is a question with many answers, which is why you might have heard of machine learning, deep learning, and possibly others if you had nothing better to do than look this stuff up.  

All of them achieve the same basic goal - making a computer learn how to solve problems - through very different methods.  Here, we'll look at what those methods really are, in the simplest possible methods, and what we can do with them.

* [Machine Learning](#machine-learning) (ML) is the process by which a computer learns and improves from experience, to make predictions or decisions, without being explicitly told how to. 

* [Deep Learning](#deep-learning) (DL) is the process by which a computer learns through representing data in the form of increasingly simplified levels.  Does that sound confusing?  Don't worry, it'll make sense when it's explained.  Probably.  DL is technically a section of ML, but it's grown so large in itself that it's worth mentioning separately.  

##  Machine Learning

Machine Learning comprises a large group of methods by which a computer can learn from data, so here, we'll look at what those methods are really doing.  

At its core, ML is about using algorithms to analyse and predict patterns in data.  If a computer is given a set of data about the heights and weights of different people, and we want to use that to predict someone's height using just their weight, then we train the computer on the data we do have, so that it can have a decent understanding of the correlation between height and weight.  How does it do that? Magic\*.

\*More accurately, Math.

Despite what some say, the math really is simple at the core.  You take a graph of variables, and plot a line connecting the points on it.  Look at this graph, for instance: 

![Data Points]( "Data Points")

If you wanted to predict what Y would be for a new value of X (e.g: 16, with the actual Y value being 14.2), how would you do it?  (Seriously, how would you do it?  I can't remember.)

So after some a few Google searches, it turns out that you draw a straight line roughly connecting all the points, and then look at what Y the new X would give on that line, and you can see it comes pretty close to the actual answer (marked with the orange dashed lines).

![Data Points]( "Data Points")

That's Machine Learning.  No seriously, that's the basic idea.  You give the computer a lot of data like this, and it "understands" it by finding patterns in them. 

### So when does it get more complicated?

Well, you might have noticed that the predicted value of Y isn't **exactly** the same as the actual value.  Now, because people are picky as hell, they want more accuracy.  So we end up with patterns like:

![Data Points]( "Data Points")

Now here, we fit exactly each point to a weird, zig-zaggy line.  But if you were looking closely, you'd see that here too, the predicted value of Y is slightly wrong.  This is because in a real-world scenario, there are more variables involved than the ones we're looking at, so individual data points may be affected in different ways, and trying to get a perfect fit will lead to errors because of the over-reliance on those individual points.  Think of it as a group of people working together toward some goal.  When each of them has less individual power, their common goal becomes more prominent, while if everyone had more power, they'd try to push things toward their own personal goals.

Thus, simply connecting the data with a straight line can get you wrong answers because you're not accurate enough, and connecting it with too perfect a fit can get you wrong answers because you're trying too hard to be accurate.  One of the real problems in Machine Learning comes in trying to hit the perfect middle.  For example, look at the following graph:

![Data Points]( "Data Points")

As you can see, here we have a curvy graph that's more flexible than a straight line, but doesn't try to be a perfect fit.  Thus, we have a predicted value of Y that's closer to the actual value than we did before.

### In the Real World

Of course, in real-world situations, there are way more than 2 variables.  The complexity of ML, and the reason it's still growing as a field, is because computer models are growing ever larger.  

A good example would be something like the recently released GPT-3, a text processing engine which uses *175 billion* parameters (they're called parameters because the method used is closer to Deep Learning, which we'll see later).  GPT-2, its predecessor, is slightly less massive with only 1.5 billion parameters, but still does some amazing, weird, brilliant stuff.  Check out [AI Dungeon](https://play.aidungeon.io/) if you want to read/write some of the craziest stories you'll ever find.  If you're familiar with Reddit, try this [subreddit filled entirely with GPT-2 bots built on different subs](https://www.reddit.com/r/SubSimulatorGPT2/).

## Deep Learning

How would you teach a baby what a "dog" is?  I doubt you'd get very far trying to explain to them all the features of a dog, and expecting them to understand.  No, you'd point to a dog, and say "That's a dog".  Now the baby associates the visual input it gets in that moment with the word "dog".  As they grow up and learn that more types of breeds are also called "dogs", their idea of what a dog is becomes more general - they learn to pick up on the unique traits that make up a dog purely from the experience of associating certain things as dogs.

That's the idea behind deep learning.  When machine learning first came to the mainstream, it was amazing.  It was powerful, and it got great results.  But people saw that it didn't quite work like human minds do.  Hence, the development of this field, where you teach a computer to associate certain characteristics with a concept through simplifying visual data into simpler forms, like you'd simplify the sight of an animal into characteristics such as "four legs", "tail", "dull claws", and simplify those further into "dog".

Look at these two pictures.  One contains just one section of solid colour, and the other contains a horizontal line separating two sections.  

Picture1                   |  Picture 2
:-------------------------:|:-------------------------:
![]()   |  ![]()

Now, how would you make a computer recognize that one has a line and the other doesn't?  Well, the images can be represented as:

| Picture 1 | Picture 2 |
|--|--|
|<table> <td>1</td><td>1</td><tr><td>1</td><td>1</td></tr> </table> | <table> <tr><td>1</td><td>1</td></tr><tr><td>0</td><td>0</td></tr> </table>|


Where 0 stands for a black pixel, and 1 for a white pixel.  

**Warning:  There is math ahead, but it's not complicated.  Really.**

Now take this new set of numbers:

|  |  |
|--|--|
| 100 | 100 |
| -100 | -100 |

If you simply multiply every number in this new table with the numbers we got from each image, what would we get?

Don't worry, you don't have to think about the math if you don't want to.  
The first picture:  100\*1 + 100\*1 - 100\*1 - 100\*1 = 0
The second picture:  100\*1 + 100\*1 - 100\*0 - 100\*0 = 200

So, the computer can use this table of numbers and receive a large number for horizontal lines in an image, thus identifying them.  In an image with many pixels, this sort of operation is performed on every small set of pixels (like the 2x2 squares we are looking at), and a smaller matrix is obtained from the operation, as every set is reduced to one number each.  For example, if the second picture above were part of a larger image, this operation performed on the image would result in that portion reducing from the 2x2 matrix above, to the number 200.

If we used a different set of numbers for the operation, say:

|  |  |
|--|--|
| 100 | -100 |
| 100 | -100 |

Then this operation would identify *vertical* lines in an image.  Like these, deep learning algorithms have many sets of numbers to identify very basic patterns in images.

This kind of operation is called a **convolution**.  In other words, a convolution is an operation done on an image to simplify it into more familiar concepts such as "horizontal line" or "circle" or the like.  

Now, from these smaller matrices we get after a convolution, we can get concepts that are composed of the previous elementary concepts; such as "box" or "wheel", with a further convolution.  During the training stage, when the computer is building the model, it learns to optimize on certain parameters.  The more parameters, the more complex the data that can be understood.  

When many convolutions like these are stacked together, a computer can, from a group of pixels, understand the presence of complicated images like a vehicle, or a person.  This cascading series of convolutions forms what's known as a neural net.

{Image}

### Other Applications

Imagine if you could apply this principle to a video game.  The computer would read the image on the screen, identify obstacles and enemies, and choose movements and actions to optimize its position almost perfectly, at every frame.  

A relatively simple example would be Flappy Bird.  We all love that game, don't we?  Only one phone of mine was broken when I played it, anyway.  A neural network trained to play Flappy Bird, would at first choose entirely random moves, then notice that if it chose to jump when a certain object moved into a certain position of the screen (such as a pipe moving close to you), it would get a higher score.  So that object is included in its network, associated with the "jump" option.  As the game keeps on training, it gets better and better at identifying ideal times to jump to increase the score.  [Some models that were trained like this kept on playing forever, because it was now too good to lose.](https://www.youtube.com/watch?v=WSW-5m8lRMs)

### In the Real World

Real-world deep learning models have many, many convolutions stacked together to be able to identify things with better accuracy and efficiency.  That face recognition feature on your phone is a deep learning model, trained to recognize human faces.  The voice assistant on your phone translates your voice into text with deep learning models.  As mentioned before, GPT-3 uses deep learning with a ridiculous number of parameters to not only understand English, but also understand story, characters, writing style, and in some cases, [game strategy](https://slatestarcodex.com/2020/01/06/a-very-unlikely-chess-game/).

### Alright, so how do I actually use this stuff?

Training and using machine learning and deep learning models is surprisingly easy, and indeed, is nothing more than a few lines of Python code that uses complicated packages that we thankfully don't have to learn anything about.  We'll cover that next time.  For now, let's look at something I think is slightly more interesting.


## Friendly AI

Now, for nearly all modern AI, there's little need to ensure that they won't accidentally exterminate humanity.  But advancements in AI are in the direction of making them broader in scope and more capable of learning and adapting to new tasks.  Thus, the necessity of researching how to align a super-intelligence's interests with our own.  Both fortunately and unfortunately, this is a relatively unexplored field.  Unlucky because it means that AI is growing at too fast a rate for us to use it safely, and lucky because it means that it's relatively less technical and mathematical.  

One of the biggest reasons as to why AI Alignment research is so crucial and time-sensitive, is because as we build AI to be smarter, at some point, the technology will reach the state where it can alter itself, or create new machines smarter than it.  This is inevitable in the future of AI research.  But the problem is, the AI is much faster and more powerful than we are.  It can achieve in seconds what we could only in years (This point, if you're interested to know more, is called the Singularity).  The first super-intelligent machine will be the last machine we ever need to make.  But once we reach that point, we cannot further research AI safety, because we would already be past the stage where it could have saved us.

Now, it's possible we're being overly scared of it.  After all, why would an AI destroy humanity?  This isn't *Terminator*, these are machines that can only do what we want them to.

Imagine we develop Delta, a super-intelligent AI.  We decide to start out safe with its tasks, and tell Delta just to increase paperclip production in a factory.  Delta, of course, immediately hijacks the world's resources and funnels them all into building more paperclips, and enslaves humanity to the task of building more paperclips.  

No matter how innocent a task may seem, no matter how many contingencies you put in, to a super-powered being whose only concern is fulfilling that task, you're ultimately only proving that your own imagination can't think of a way to beat those contingencies and achieve the task better.  

The two fundamental problems features of any super-intelligence that lead to failure in commonly suggested methods for Friendly AI are

* Power: The AI is far more powerful than we can even comprehend, and can thus achieve its goals with far more efficient methods than we could dream of.  This means that we cannot possibly try to plan for what an AI might do.
* Literalness: The AI only cares about the task given to it, not the mindset behind that task.  If it's told to cure cancer, it wouldn't understand that what we really mean is the well-being of humanity, it would just enslave humanity to test on and find a cure better.  Goals like "maximizing human happiness" sounds good, until the AI straps everyone to a chair and pumps them full of dopamine till they die.

Then why not imprison the AI in a box, and only allow it to give us answers for our questions, and not impact the world in any other way?  The problem is that again, an AI can easily convince a human to find a way to let it out (because the AI sees it being free as a faster way to fulfil its task) even with just plain text.  The [experiment was actually conducted with two people](https://yudkowsky.net/singularity/aibox/), and the first two times, the person in the box actually talked their way out.  So obviously, it wouldn't be a problem for an AI to do it.

How about we use some of that Machine Learning, and train the AI to learn what our sense of morality is, and then only take choices that we would too?  There are two problems with that, the latter of which will also prevent most solutions like these.  The first, and a more solvable issue, is that common human morality is often flawed - between the choice of saving one person dying in front of you, and five dying somewhere you can't see, most people choose the one in front of them, due to empathy.  Which is relatively harmless on a scale like this, but we wouldn't want the AI to save one billion lives while letting the rest of humanity die.

The second problem, and one of the primary reasons the problem of AI Alignment is so difficult, is that whatever restrictions we place on the AI's behaviour, are all things it is motivated to try and change, especially once it reaches the level of power where it can alter itself.  It might even be able to create a new AI, which aren't affected by the same restrictions.  This is the same reason why simply programming an AI to not harm us, or to follow Asimov's Laws of Robotics would not work.

If you're still reading at this point, and I'm grateful (and honestly, surprised) if you are, here's a [link](https://intelligence.org/ie-faq/#FriendlyAI) that goes into further detail (while still being incredibly interesting to read, I swear) into the nature of Friendly AI.