# Part the 4th - Artificial Intelligence

> AI is whatever hasn't been done yet.
>
> \- Larry Tesler

Our bodies are shells for a small, squishy, slimy lump of gray, white, and red.  The kind of thing that if you saw on the side of the road, you'd go out of the way to avoid.  It looks like the most damage it could do to you is your own disgust.

Humanity did not become the most powerful species on the planet because we had the strongest muscles, the sharpest claws, or the toughest hides.  Millions of years ago, the rulers of the land were animals who were really good in those niches, and really, those niches alone - what could travel by day and night, could travel through the land, water, and skies, or even just have a good offence and defence at the same time?  Well, as it turns out, the small squishy lumps.  No armour, no claws, no poison, and somehow, able to weather every climate, cut diamonds, and cure plagues.

Now imagine someone smarter than us.

It's hard to picture what exactly AI can do.  The idea of someone curing cancer, or visiting Mars, is evocative, a powerful image.  But if someone told you there was one single technique that could cure every disease, take us to Mars, to Jupiter, to Alpha Centauri, that could create the grand unified theory, end hunger, end ageing, make everyone intelligent, crack the genetic code - you get the point - it'd just sound like they were trying to sell you something.

But we do already have one thing that cured smallpox and took us to the Moon and created science.  Sure, we don't fully understand what that thing is yet, but that's our fault, not some quality of intelligence that makes it not-understandable.  Lightning seemed like it was beyond our understanding and control before electricity.  And our understanding of intelligence only increases over time.  And that which we understand, we can create.

There's a quote from Eliezer Yudkowsky, one of the largest names in AI Safety (and one you'll see in this part often), that applies here:
> There’s a popular concept of “intelligence” as book smarts, like calculus or chess, as opposed to say social skills. So people say that “it takes more than intelligence to succeed in human society”. But social skills reside in the brain, not the kidneys. When you think of intelligence, don’t think of a college professor, think of human beings; as opposed to chimpanzees.

Intelligence is the source of technology.  What happens when we use that technology to better our intelligence?  We could create better scientific theories, medicines, movies, but most of all, we could use it to take our intelligence even further.  That's a cycle with an upper limit so far away we can't see it yet, or even know if it exists.  It's a runaway intelligence explosion, better known as the Singularity.  While it's instinctive to think of the range of intelligence as stretching from a flat-earther to Einstein, it actually goes more like:
‎

![Intelligence chart](https://imgur.com/d4aTBrR.png)
‎

When you consider this, the next few decades become quite probably the most important in human history and its future.

## What's Intelligence?

> By far the greatest danger of Artificial Intelligence is that people conclude too early that they understand it.
>
> \- Eliezer Yudkowsky

I usually leave the recommendation for further reading to the end, but in this case, it's important enough that it has to be said in the beginning: Godel Escher Bach is the greatest book ever written on the nature of intelligence, one of the greatest books *ever*, and a very entertaining read about music, art, philosophy, surrealism, literature, programming - you name it, Douglas Hofstadter's probably written about it. If you came up to me with no prior knowledge about *anything* related to AI but wanted to get into it, this is what I'd give you (well, not *give* you because I probably wouldn't trust you with my copy).

### Human Bias

The first task when we're dealing with knowing more about intelligence is the very *human* traits that we unknowingly think of as a property of all intelligence.  The human mind is a literal smorgasbord of different parts that were hacked together through slow evolutionary building.  What we picture when we think of intelligence is really just an optimization process that's mixed with a lot of other things, like ideas about what to optimize for (morality, etc) and things that negatively affect that optimization process (emotions like worry, fear, etc).  When we make an AI, it wouldn't look anything like us.

When you think of AI destroying humanity, you probably think of Skynet.  After all, Terminator is the most famous representation of evil artificial intelligence in media.  But Arnold Schwarzenegger is not what an AI would be like.  He's not even *close*.  Nor is Ultron, or Agent Smith.  HAL 9000 from 2001: A Space Odyssey and Ava from Ex Machina are closer to reality, but even then only relatively.

An unfriendly AI would not 'rise up' against humanity because it dislikes being controlled by those it views inferior.  Remember, intelligence, in its purest form, is nothing more than an optimization process for a goal  (Humans only seem more complicated because we have more than a few goals at the same time, and our optimization process isn't that great.)  What an AI wants, what it likes and dislikes, are all things we define for it.

An AI would not feel the emotion humans describe as 'hate' is, unless we specifically program that in (which, by the way, isn't just monstrously stupid, but far more difficult than it seems - emotions are very, very complex mechanisms, and while it's possible to program it, it just isn't worth the effort.)  But even beyond that, the form our thoughts take are largely a product of the specific evolutionary process we've gone through as a species to get here; an AI would simply think in different ways.  For example, can you imagine what it would be like to think in multiple parallel lines of thought, the way even old processors function?

Finally, the last thing (roughly, there's a lot to be said about anthropomorphic bias when dealing with AI, but we only have so much space) is that AI will be better than us.  Not just faster, but able to think in more efficient ways, and use that power in ways that we probably wouldn't be able to think of, limited by very specifically defined biological pathways that we are, and able to self-modify as they are.  AI is not a case of 'book-smarts' (really good at things you usually think of when you hear 'intelligent', like chess or math), but one that's better than us at everything - social skills, military strategy, even musical skill.  Those are all things that come from our brains, after all.  

Remember when we talked about the AI in a box idea, where we imprison the potentially dangerous AI inside a system it cannot escape?  Yeah, look at that intelligence chart above, and ask yourself how many humans would be able to convince you of something you don't want to do.  Then ask yourself how much further down the line they would have to be to be able to do it (If you're a theist, ask yourself whether your god could convince you).

### Emergence and Strange Loops

The neurons in your brain are complex things, but with a very simple function.  A single neuron has a few synapses ('entry ports') and one axon ('output port').  It takes in multiple inputs, sums them, and if the sum is greater than a certain value (some of them can be negative) , it fires a signal (send ions) down the axon.  That's all there is to the lowest level of the mind - a simple addition function.  So where does the complicated stuff that makes up intelligence come in?

There is a concept called *emergence*, which goes by many definitions depending on whom you ask, but a more-or-less accurate one is this:  Emergence is when small parts with simple functions combine to form larger parts with more complex functions (like those neurons combining to form you); those large parts would interact with each other in ways the smaller parts do not (like you reading this, while a neuron probably cannot read something another neuron wrote).  

Most definitions for this word are accurate to an extent, but the one that describes intelligence as something more than the sum of its parts is one that's technically correct, but greatly misunderstood.  It is not that neurons combining together mysteriously creates functions that come from somewhere outside the neurons, it's simply that a random combination of neurons would not create intelligence, and the process of co-ordinating them in highly specific ways is the *more* that is given to the parts to form intelligence.

Think about an AI algorithm running on a computer.  At its core, it breaks down to a lot of transistors going on and off.  Sequences of instructions in this binary language are combined to form logical gates, and then functions like add and move.  Sequences of these functions are then combined to form operations we're more familiar with, like the code that we use to write the algorithm (Python), and the various math we implement using it.  At no point in this process is there anything *more* than switches going on and off, but the way they interact allows for increasingly complex things to happen.

Likewise, a very small group of neurons arranged in a specific way can handle more complexity than a single neuron, as they now have a number of internal neurons that can process the input signals between them before giving more than a single output signal (which is what a single neuron does).  These groups can then combine to form larger meta-groups, and those to form meta-meta-groups, and so on.  

These groups (and meta-variants) are almost always temporary, in that they form all over the brain as necessary, and do not need a specific set of neurons to do it - any will do (neurons don't feel anger at someone taking their job).  A neuron can likewise be a part of many groups at once, which is where it becomes so complicated you might be beginning to understand why we still haven't completely understood it yet.

The *self*, by that same logic, is just a very, very high-level group that's so complex it can't easily understand the different parts that form it - in the same way we wouldn't expect an ant to understand its own behaviour, despite it being simple enough that *we* understand it.  Douglas Hofstadter describes the specific structure of neurons that form the consciousness as a *strange loop*, a hierarchy of levels in which each level is connected to each other in a large, tangled mess that loops back on itself (if you don't understand that - and I'm a little worried if you did - look at *Godel Escher Bach*).  Memories, creativity, personal unique ways of thinking - these are all mind-numbingly complex groups of billions of neurons.

## AI Alignment

> The AI does not hate you, nor does it love you, but you are made out of atoms which it can use for something else.
>
> \- Eliezer Yudkowsky

### Self-modifying AI

So if all our skills are a product of our intelligence, and we progress our technology to the point where we create an intelligence exactly as powerful as us, then it would be as capable as we are at technology, right?  So it would be just as good as we are in designing AI systems that are more advanced than before?  And then *that* AI would be even better at creating more advanced AI, and so on, until in a very short amount of time to us, we've gone from a human-level robot friend to a minor god-level being, and it's *still* increasing.

If that scenario seems slightly concerning, consider what it would be like if we haven't properly aligned with our interests.  We can't just try to control it, because it'll blow past our wildest imaginations faster than we can do anything about it.  If we haven't come up with a way not to make the AI unable to harm us, but to not *want* to harm us (more on that in the following section), then it's lights out.  That kind of AI would be a [paperclip maximizer](https://wiki.lesswrong.com/wiki/Paperclip_maximizer), an intelligence with a very high optimization power, but without a well-calibrated sense of what we want that power to be used for.  In the example in that hyperlink, we'd end up with every resource in the known universe gradually being converted into paperclip manufacturing factories, as humanity is instantly wiped out to prevent any chance of that goal being hindered (*that's* the realistic scenario of why an AI might destroy humanity), and paperclip factories just aren't fun enough to be worth that.

Sure, we could technically not have the AI self-modify, but in a world where technology is freely available and everyone's looking for a way to get ahead without care for the consequence, who's going to stop every last person from using the obvious most efficient method to improve AI?

### Value Systems

We know that an AI is better than us in every way - there's no movie third act scene where we realize that machines will never understand creativity or love and that we can use that to beat it; if an AI is smarter than us, it understands *those things* better too.  So when you look at it that way, the idea of controlling an AI seems extremely difficult, right?  *Right*?

Well yes.  Anything we're intelligent to come up with for an AI will be blown apart by it in a couple of hours after it reaches our level and is made to self-modify.

### Making the AI do all the work <AI monitoring other AI>

### Storytime - The AI in a Box Boxes You