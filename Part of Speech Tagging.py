#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk

from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')

size = int(len(brown_tagged_sents) * 0.9)

train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

print(train_sents)
print(test_sents)
#train the tagger
unigram_tagger = nltk.UnigramTagger(train_sents)

#calculate the accuracy
print("Results on test set {0}".format(unigram_tagger.evaluate(train_sents)))
print("Results on test set {0}".format(unigram_tagger.evaluate(test_sents)))


#1) Why is the training accuracy higher than the testing accuracy?
# in practice if data sets are good and clear, the score will be higher. If the data set
# is not good like null values or unknown words then the scoring model is less percentage. 
#in this example the training data sets. There are no unknown words. When print(train_sents)
#there is less unkown words and the scoring model is better. 

#2) Why is the training accuracy not perfect (100%)
#in the test model, after print(test_sents) there are a lot of unknown words and null values,
#so the scoring model is less percentage compared to training data set.


def_tagger= nltk.DefaultTagger("NN")
uni_tagger=  nltk.UnigramTagger(train_sents, backoff=def_tagger)

print("Results on test set {0}".format(uni_tagger.evaluate(train_sents)))
print("Results on test set {0}".format(uni_tagger.evaluate(test_sents)))



# In[4]:


def_tagger= nltk.DefaultTagger("NN")
uni_tagger=  nltk.UnigramTagger(train_sents, backoff=def_tagger)

print(test_sents)
print(train_sents)

print("Results on test set {0}".format(uni_tagger.evaluate(train_sents)))
print("Results on test set {0}".format(uni_tagger.evaluate(test_sents)))

#3 Why does the accuracy score on the training data not go up but it does on the test data?
#The accuracy score on training did not change because the size of the data is smaller and the
#data has fewer unknown words and less null values. In the testing data there was more data
#size thus improving the accuracy score by 2 points. 


# In[25]:


size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

BGT0 = nltk.DefaultTagger('AT')
BGT1 = nltk.UnigramTagger(train_sents, backoff=BGT0)
BGT2 = nltk.BigramTagger(train_sents)
#BGT3 = nltk.BigramTagger(train_sents)

print(test_sents)
print(train_sents)


print(BGT2.evaluate(test_sents))
print(BGT1.evaluate(test_sents))
#4 Create two new taggers, A BigramTagger that has not backoff and a 
#BigramTagger that user a unigram tagger as backoff. Report the accuracies. 
#Why is one so much lower than the other?
#because in backoff tagging, you are combining the unique powers of each tagger 
#in order to build an algorithm which makes the best possible decisions thus 
#improving the accuracy score. 



# In[24]:


#5 Repeat #4 with a TrigramTagger using a Bigramtagger as backoff
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

BGT0 = nltk.DefaultTagger('AT')
BGT1 = nltk.BigramTagger(train_sents, backoff=BGT0)
BGT2 = nltk.TrigramTagger(train_sents)
#BGT3 = nltk.BigramTagger(train_sents)

print(test_sents)
print(train_sents)


print(BGT2.evaluate(test_sents))
print(BGT1.evaluate(test_sents))


# In[ ]:




