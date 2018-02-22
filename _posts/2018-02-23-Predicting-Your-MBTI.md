---
layout: post
title: "Predicting Your MBTI"
date: 2018-02-23
---

# Predicting MBTI types using text alone

## A brief introduction

MBTI, short for Myers-Briggs Type Indicator, is a personality metric developed by Katharine Cook Briggs and her daughter Isabel Briggs Myers, based on Carl Jung's theory on psychological types. Today, it is a common tool used by individuals and organizations alike, be it to better understand themselves or to optimize workplace dynamics.

The most common way of finding out of our type is to visit free personality test websites, where they would require you to answer questions after questions after questions (or statements) in order to determine your type, as accurately as possible. More often than not, these questions relate directly to the type characteristic which requires you to rate how well you 'relate' to the question asked. For example:

> 25. My idea of relaxation involves reading a book by the beach
> 
 * [] Strongly Disagree
 * [] Disagree
 * [] Neutral
 * [] Agree
 * [] Strongly Agree

If you haven't noticed yet, this presents a series of problems, in no order of magnitude:

1. We'd inherently know that this question refers to Extroversion/Introversion, and hence may tend to answer based on how we identify rather than purely relating to the question asked.
    * In other words, we'd answer with some form of a bias i.e. Who we think we are or want to be vs who we really are.
    * Come to think of it, regarding the spelling of extrovert or extravert (oh good lord spellcheck got activated for the latter!), I found this.
2. I identify as a strong introvert but I don't really dig reading a book by the beach, nor do people from landlocked areas etc...
3. We cannot, in some senses, identify by how much we agree/disagree with the statement/question. However, in order for the model to work, we have to choose a side. Strongly.
4. Answering so many questions is already by itself a big time waster, not to mention it being a tiring process.

The question wasn't picked up from any site in particular by the way, I made it up!

## Proposal

My project shall attempt to aid users in having a seamless experience in finding out their MBTI type. Instead of having the user dedicate his/her precious time and brain energy to processing all the questions, the machine only needs to pick up the existing messages produced by the user to predict their MBTI type!

Read on if you would like to understand the how, but beware, it can get a little technical. Otherwise, click here to go straight to the webapp for some fun!

### How
The model makes use of forum posts from personalitycafe.com for training. Luckily for me, this dataset is already made available on Kaggle in the form of 50 posts per person of a certain MBTI type. Not a competition piece though, just a dataset for us to play around with.

The dataset only consists of two columns, one being the type and the other being the collection of posts made by the person of the type. As such, therein begins my journey of digging out and processing the data from the text followed by modelling and predicting.

Here is a preview of the first line:

--------------------------------------------

type: 
``` 
INFJ
```

posts: 
``` 
"'http://www.youtube.com/watch?v=qsXHcwe3krw|||http://41.media.tumblr.com/tumblr_lfouy03PMA1qa1rooo1_500.jpg|||enfp and intj moments https://www.youtube.com/watch?v=iz7lE1g4XM4 sportscenter not top ten plays https://www.youtube.com/watch?v=uCdfze1etec pranks|||What has been the most life-changing experience in your life?|||http://www.youtube.com/watch?v=vXZeYwwRDw8 http://www.youtube.com/watch?v=u8ejam5DP3E On repeat for most of today.|||May the PerC Experience immerse you.|||The last thing my INFJ friend posted on his facebook before committing suicide the next day. Rest in peace~ http://vimeo.com/22842206|||Hello ENFJ7. Sorry to hear of your distress. It's only natural for a relationship to not be perfection all the time in every moment of existence. Try to figure the hard times as times of growth, as...|||84389 84390 http://wallpaperpassion.com/upload/23700/friendship-boy-and-girl-wallpaper.jpg http://assets.dornob.com/wp-content/uploads/2010/04/round-home-design.jpg ...|||Welcome and stuff.|||http://playeressence.com/wp-content/uploads/2013/08/RED-red-the-pokemon-master-32560474-450-338.jpg Game. Set. Match.|||Prozac, wellbrutin, at least thirty minutes of moving your legs (and I don't mean moving them while sitting in your same desk chair), weed in moderation (maybe try edibles as a healthier alternative...|||Basically come up with three items you've determined that each type (or whichever types you want to do) would more than likely use, given each types' cognitive functions and whatnot, when left by...|||All things in moderation. Sims is indeed a video game, and a good one at that. Note: a good one at that is somewhat subjective in that I am not completely promoting the death of any given Sim...|||Dear ENFP: What were your favorite video games growing up and what are your now, current favorite video games? :cool:|||https://www.youtube.com/watch?v=QyPqT8umzmY|||It appears to be too late. :sad:|||There's someone out there for everyone.|||Wait... I thought confidence was a good thing.|||I just cherish the time of solitude b/c i revel within my inner world more whereas most other time i'd be workin... just enjoy the me time while you can. Don't worry, people will always be around to...|||Yo entp ladies... if you're into a complimentary personality,well, hey.|||... when your main social outlet is xbox live conversations and even then you verbally fatigue quickly.|||http://www.youtube.com/watch?v=gDhy7rdfm14 I really dig the part from 1:46 to 2:50|||http://www.youtube.com/watch?v=msqXffgh7b8|||Banned because this thread requires it of me.|||Get high in backyard, roast and eat marshmellows in backyard while conversing over something intellectual, followed by massages and kisses.|||http://www.youtube.com/watch?v=Mw7eoU3BMbE|||http://www.youtube.com/watch?v=4V2uYORhQOk|||http://www.youtube.com/watch?v=SlVmgFQQ0TI|||Banned for too many b's in that sentence. How could you! Think of the B!|||Banned for watching movies in the corner with the dunces.|||Banned because Health class clearly taught you nothing about peer pressure.|||Banned for a whole host of reasons!|||http://www.youtube.com/watch?v=IRcrv41hgz4|||1) Two baby deer on left and right munching on a beetle in the middle. 2) Using their own blood, two cavemen diary today's latest happenings on their designated cave diary wall. 3) I see it as...|||a pokemon world an infj society everyone becomes an optimist|||49142|||http://www.youtube.com/watch?v=ZRCEq_JFeFM|||http://discovermagazine.com/2012/jul-aug/20-things-you-didnt-know-about-deserts/desert.jpg|||http://oyster.ignimgs.com/mediawiki/apis.ign.com/pokemon-silver-version/d/dd/Ditto.gif|||http://www.serebii.net/potw-dp/Scizor.jpg|||Not all artists are artists because they draw. It's the idea that counts in forming something of your own... like a signature.|||Welcome to the robot ranks, person who downed my self-esteem cuz I'm not an avid signature artist like herself. :proud:|||Banned for taking all the room under my bed. Ya gotta learn to share with the roaches.|||http://www.youtube.com/watch?v=w8IgImn57aQ|||Banned for being too much of a thundering, grumbling kind of storm... yep.|||Ahh... old high school music I haven't heard in ages. http://www.youtube.com/watch?v=dcCRUPCdB1w|||I failed a public speaking class a few years ago and I've sort of learned what I could do better were I to be in that position again. A big part of my failure was just overloading myself with too...|||I like this person's mentality. He's a confirmed INTJ by the way. http://www.youtube.com/watch?v=hGKLI-GEc6M|||Move to the Denver area and start a new life for myself.'"
```
-------------------------------------------

Wow. Imagine 8675 times that chunk of text!


### TL;DR
If, at this point, you are already feeling afraid of what is to come, here is pretty much a summary of what I did with the dataset:

* Basic data cleaning (or the lack thereof - its actually quite clean already)
* Extract weblinks from each user, and
  * Further divide the weblinks into video, images and others
  * For videos, extract video titles from the respective sites (not used)
  * For images, extract keywords (not done)
  * For other websites, obtain categories (not done)
* Split the target data from 16 categories into 4 binary classifiers (The why will be explained below)
* Extract other metadata:
  * Emoticons (eg. :happy:)
  * Mentions (eg. @tamagonii)
  * Hashtags (eg. #ilovetamago)
  * MBTI reference (eg. INFP, ENFJ) (not used)
  * Action words (eg. *jumps into the pool and swim away*)
  * Enneagram regerence (eg. 4w1) (not used)
  * Brackets (eg. [quote])
  * Dots count (...)
  * Number of words
  * Number of word characters
  * Number of fully capitalized words (eg. HEY Y'ALL!!)
  * Number of characters of fully capitalized words
  * Ratio of fully capitalized words vs all words
  * Ratio of characters of fully capitalized words vs characters of all words
  * Median number of words used per person
  * Median number of characters used per person
* Perform Parts-of-speech (POS) tagging to the word document
* For each MBTI type,
  * Perform Term Frequency - Inverse Document Frequency (TF-IDF)
    * For word range of 1-3, up to 10,000 words/phrases each
    * For each word, apply Truncated Singular Value Decomposition (Truncated SVD) to reduce the size to 500 features each, totalling 1500 features
  * Pre-process the data:
    * Perform Standard Scaling on metadata
    * Combine with TFIDF data, and perform MinMax Scaling
  * Select 100 best features using chi2 test
  * Train the model with Logistic Regression
* Collect instances of all 4 types to predict data from new input data
* Done!

If you haven't noticed already, I used some 'big' words to describe the process. If you would like to learn what they mean and what they do, read on below!

## The ~~boring~~ important stuff.
