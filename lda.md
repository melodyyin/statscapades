---
title: "Latent Dirichlet allocation with beauty reviews"
author: "Melody"
date: "August 9, 2015"
output: html_document
---

A few weeks ago, I scraped some comments from [MakeupAlley](www.makeupalley.com) (MUA), which is not only the largest but also the most trusted beauty reviews database. For most beauty-related products, "*item name* makeupalley" is one of the top suggested queries on Google. 

I used Python and **BeautifulSoup** to scrape the reviews of 10 random items, each with around 100 reviews each and I loaded them in a txt file. The resulting dataset is quite small (~1MB), but I thought it was a good size for playing around with a text analysis model I had heard about on one of my favorite podcasts (shoutout to [Talking Machines](http://www.thetalkingmachines.com/)) - Latent Dirichlet allocation (LDA). The R packages I am using are **tm**, **slam** and **topicmodels**. I found the [topicmodels vignette](https://cran.r-project.org/web/packages/topicmodels/vignettes/topicmodels.pdf) to be especially helpful. 

In a nutshell, given a set of documents, LDA can give you the categories of words (i.e., topics) that each document is composed of. For example, a news article could be 80% politics and 20% health. This is useful because if you have a massive amount of unlabeled documents, you can use LDA to help get clues of what each document is about. 

~

Let's take a look at the 10 selected items and read in the comments: 


```r
names = readLines("names.txt")
comments = readLines("comments.txt")
unique(names)
```

```
##  [1] "The Body Shop Tea Tree Skin Clearing Toner"                   
##  [2] "MAC Sheertone Shimmer Blush - Ambering Rose"                  
##  [3] "Maybelline Baby Lips Dr. Rescue"                              
##  [4] "LUSH The Olive Branch"                                        
##  [5] "Dior Diorskin Star"                                           
##  [6] "Wen by Chaz Dean Fig Cleansing Conditioner"                   
##  [7] "Dove Sensitive Skin Body Wash with NutriumMoisture, Unscented"
##  [8] "Wet 'n' Wild Silk Finish Blush - Mellow Wine 833D"            
##  [9] "Smashbox Camera Ready Full Coverage Concealer"                
## [10] "Tom Ford Black Orchid EDP"
```

Then, load **tm** and do some basic processing: 


```r
library(tm)
c.corpus = Corpus(VectorSource(comments))
c.dtm = DocumentTermMatrix(c.corpus, control=list(stopwords=TRUE, minWordLength=3, removeNumbers=TRUE, removePunctuation=TRUE, tolower=TRUE, stemDocument=TRUE))
dim(c.dtm) 
```

```
## [1] 1042 7540
```

Find the total frequency-inverse document frequency (tfidf), which is the term frequency weighted by the log of number of documents over the number of documents that the word appears in. Basically, it is a metric that indicates how important a word is to a document. We then remove the bottom quartile of words to remove the words that appear infrequently or have little meaning. 


```r
library(slam)
c.tfidf = tapply(c.dtm$v/row_sums(c.dtm)[c.dtm$i], c.dtm$j, mean) * log2(nDocs(c.dtm)/col_sums(c.dtm>0))
c.dtm = c.dtm[,c.tfidf >= quantile(c.tfidf, 0.25)] # get rid of irrelevant terms 
dim(c.dtm) 
```

```
## [1] 1042 5679
```

Now, the actual LDA procedure using Gibbs sampling, discarding the first 100 samples and taking every 100th sample from there until the algorithm has been repeated 2000 times: 


```r
library(topicmodels)
k = 10 # for the 10 products
sd = 2015 
c.tm = LDA(c.dtm, k=k, method="Gibbs", control=list(seed=sd, burnin=100, thin=100)) # default 2000 iterations
terms(c.tm, 5)
```

```
##      Topic 1     Topic 2      Topic 3   Topic 4     Topic 5    
## [1,] "body"      "foundation" "apply"   "blush"     "scent"    
## [2,] "wash"      "coverage"   "lasts"   "beautiful" "fragrance"
## [3,] "toner"     "concealer"  "problem" "brush"     "strong"   
## [4,] "acne"      "shade"      "amazing" "pigmented" "black"    
## [5,] "sensitive" "perfect"    "every"   "light"     "dark"     
##      Topic 6      Topic 7        Topic 8    Topic 9       Topic 10
## [1,] "pretty"     "shower"       "out"      "hair"        "lips"  
## [2,] "feels"      "scent"        "worth"    "wen"         "baby"  
## [3,] "repurchase" "smells"       "anything" "conditioner" "lip"   
## [4,] "all"        "moisturizing" "thought"  "fig"         "colour"
## [5,] "smooth"     "gel"          "nothing"  "stuff"       "balm"
```

How cool! The algorithm was able to discern the various product types that the 10 items belong to. Topic 1 is about toner (item 1), topic 2 is about face makeup (items 5 and 9), topic 4 is about blush (items 2 and 8), topic 5 is about fragrance (item 10), topic 7 is about body wash (items 4 and 7), topic 9 is about hair conditioner (item 6), and topic 10 is about lip product (item 3). All items are accounted for. Topics 3, 6 and 8 are a little vague and don't seem to lean towards any product category, although this is expected since some items belong in the same category. They could be reactions to these items, however; 3 and 6 sound positive, while 8 is more neutral. 

Here are the results for slightly fewer topics (k=7): 


```
##      Topic 1 Topic 2      Topic 3     Topic 4  Topic 5      
## [1,] "toner" "foundation" "blush"     "scent"  "hair"       
## [2,] "acne"  "coverage"   "brush"     "smells" "wen"        
## [3,] "tree"  "concealer"  "light"     "strong" "conditioner"
## [4,] "tea"   "shade"      "pigmented" "black"  "fig"        
## [5,] "again" "perfect"    "apply"     "dark"   "stuff"      
##      Topic 6        Topic 7 
## [1,] "body"         "lips"  
## [2,] "shower"       "baby"  
## [3,] "wash"         "feels" 
## [4,] "sensitive"    "lip"   
## [5,] "moisturizing" "pretty"
```

It appears that those reaction categories have been removed. Finally, we try using 3 topics: 


```
##      Topic 1      Topic 2     Topic 3
## [1,] "foundation" "hair"      "lips" 
## [2,] "blush"      "scent"     "body" 
## [3,] "coverage"   "wen"       "wash" 
## [4,] "concealer"  "shower"    "toner"
## [5,] "shade"      "fragrance" "acne"
```

It's reasonable to name the first topic as makeup and the third as skincare. The second is a little difficult.. perhaps personal hygiene?

Even though I know LDA is used more for determining the topic distribution of a document rather than topic discovery (i.e., we care about the distribution of the top categories per document rather than the single most prevalent topic), it's awesome that the algorithm was able to discern the different product types given very limited set of documents and only the number of categories to look for. 

Let's show an example of the topic distributions (using k=3) of a random comment:


```r
set.seed(809)
choice = sample(1:1042, 1)
comments[choice]
```

```
## [1] "I have dry/combination skin and I get breakouts often and I LOVE this toner!!! I had been using the Clinique Clarifying Lotion #2 (which is just a fancy name for a toner) and it gave me odd bumps on my forehead. I decided to switch back to this toner a few weeks ago and the bumps went away over night. I'm not sure if I'll purchase this toner again or another one from The Body Shop (i've heard good things about the Vitamin E toner so I might give it a try...). This is a very good price in my opinion, only $11 and The Body Shop often has sales. I haven't noticed this drying out my skin at all but I do use a good moisturizer (h20 plus face oasis hydrating treatment). Overall I love this and I won't be hesitant to purchase again!"
```

```r
posterior(c.tm)$topics[choice,]
```

```
##         1         2         3 
## 0.2271062 0.2161172 0.5567766
```

56% skincare, 23% makeup, 22% personal hygiene; this was a review for a toner. 

There were similar results with higher k-value; although the correct category would have the highest value, the rest tended to be all similar. This suggests that perhaps LDA works better on longer documents with a greater set of vocabulary or with a larger set of documents. Or, alternatively, my dataset is not appropriate for LDA analysis since the majority of comments will hover around one or two topics. Nevertheless, LDA is a great algorithm to have in my toolkit and I will certainly come back to this algorithm if I encounter any data that would be suited for its usage! 
