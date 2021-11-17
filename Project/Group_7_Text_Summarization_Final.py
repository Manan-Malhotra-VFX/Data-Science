#!/usr/bin/env python
# coding: utf-8

# # Basic Imports and Installations

# In[1]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


get_ipython().system('pip install PyPDF2')
get_ipython().system('pip install flair')
get_ipython().system('pip install bert-extractive-summarizer')
get_ipython().system('pip install torch')
get_ipython().system('pip install sumy')
get_ipython().system('pip install afinn')
get_ipython().system('pip install rouge')
get_ipython().system('pip uninstall pandas')
get_ipython().system('pip install pandas')
# !pip3 install --upgrade pandas


# In[130]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.express as px

import PyPDF2
import datetime as date
import re
import string
import spacy
import nltk
import math
import torch
# import json

from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest
from string import punctuation
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk import PorterStemmer
from afinn import Afinn
from rouge import Rouge
from wordcloud import WordCloud, STOPWORDS

from gensim.summarization import summarize, keywords
# import os
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.lex_rank import LexRankSummarizer 
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import get_stop_words
#from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import pickle
# import requests
from textblob import TextBlob


# In[131]:


# creating a pdf file object to read pdf file
#pdf_file_obj = open('/content/gdrive/MyDrive/Project/Ecommerce_Business_Guide.pdf','rb')
pdf_file_obj = open('Ecommerce_Business_Guide.pdf','rb')


# In[132]:


#creating pdf filereader object
pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)


# In[133]:


def displayInfoBook(pdf_reader):
    information = pdf_reader.getDocumentInfo()
    print("Author:",information.author)
    print("Creator:",information.creator)
    print("Producer:",information.producer)
    print("Subject:",information.subject)
    print("Title:",information.title)
    print("Number of Pages:",pdf_reader.getNumPages())
      
displayInfoBook(pdf_reader)


# In[134]:


no_pages = pdf_reader.getNumPages()

start = date.datetime.now()
corpus = ''
for i in range(0, no_pages):
    page = pdf_reader.getPage(i)
    corpus += page.extractText()

end = date.datetime.now()
pdf_file_obj.close()

print("Time taken =",end-start)
print("length of corpus =",len(corpus))
print(corpus)

ds                   = pd.DataFrame()
ds['Corpus Length']  = pd.DataFrame([[len(corpus)]])
ds['Corpus']         = pd.DataFrame([[corpus]])


# # Data Cleaning/ Preprocessing

# In[135]:


corpus = corpus.replace("'s",'') # replaces apostrophe s
corpus = corpus.replace('\n','') # replaces newline character
corpus = re.sub(r'\([^()]*\)','',corpus) # removes text inside brackets including brackets
corpus = re.sub(r'(http|https|www)\S+', '', corpus) # replaces www.digitalsherpa.com,http://www.articlesbase.com/technology
corpus = re.sub(r'\<.+\>','',corpus) # replaces <link rel=ﬂcanonicalﬂ href=ﬂﬂ />
corpus = re.sub(r'\s+',' ',corpus) # replaces more than 2 spaces with 1 space
corpus = corpus.lower() # converts the text to lower

ds['Cleaned Corpus']         = pd.DataFrame([[corpus]]) 
ds['Cleaned Corpus length']  = pd.DataFrame([[len(corpus)]])

print("corpus length =",len(corpus))
corpus


# In[136]:


#dataset
ds


# # Exploratory Data Analysis(EDA)

# #### Using Spacy & NLTK

# In[137]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# In[138]:


#creating list of sentences
list_sentences = sent_tokenize(corpus)
print("Total no of sentences = ",len(list_sentences),"\n")
list_10 = list_sentences[0:10]
list_10


# In[139]:


#Named entity recognition for sentences between 80 to 88
ner = spacy.load('en_core_web_sm')

for i in range(80,88):
    one_sent = list_sentences[i]
    doc_block = ner(one_sent)
    displacy.render(doc_block, style='ent',  jupyter=True)    


# In[140]:


#Named entity recognition for entire text

colNames = ['TEXT', 'LABEL', 'MEANING'] 
Entities_table = pd.DataFrame() #initialising dataframe

ner = spacy.load('en_core_web_sm')
ner_data = ner(corpus)
for word in ner_data.ents:
    varLabelValue =  word.label_
    varLabelMeaning = spacy.explain(varLabelValue)
    Entities_table = Entities_table.append(pd.DataFrame(data=[[word.text, word.label_, varLabelMeaning]],
                                                        columns = colNames))
    #print(word.text, word.label_, word.lemma_)
    
    
Entities_table.reset_index(drop=True)


# * Text: The original word text.
# * Lemma: The base form of the word.
# * POS: The simple UPOS part-of-speech tag.
# * Tag: The detailed part-of-speech tag.
# * Dep: Syntactic dependency, i.e. the relation between tokens.
# * is alpha: Is the token an alpha character?
# * is stop: Is the token part of a stop list, i.e. the most common words of the language?

# In[141]:


# POS and Dependency on entire corpus (string)
#creating table with all values
#creating noun and verb dictionary with key as text and values as their frequency

colnames = ['TEXT', 'LEMMA', 'POS','TAGS','DEP','ALPHA','STOP']
tags_pos_table = pd.DataFrame()

noun_frequencies = {} # dictionary intialization for noun
verb_frequencies = {} # dictionary intialization for verb

nlp = spacy.load('en_core_web_sm')
ner_data = nlp(corpus)
for token in ner_data:
    tags_pos_table = tags_pos_table.append(pd.DataFrame(
        data=[[token.text,token.lemma_,token.pos_, token.tag_, token.dep_,token.is_alpha,token.is_stop]],
                         columns = colnames))
        #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.is_alpha, token.is_stop)
    
    #create dictionary of noun frequencies
    if token.pos_ == 'NOUN':
        if token.text not in noun_frequencies.keys():
            noun_frequencies[token.text] = 1
        else:    
            noun_frequencies[token.text] +=1 
    
    #creates dictionary of verb frequencies        
    if token.pos_ == 'VERB':
        if token.text not in verb_frequencies.keys():
            verb_frequencies[token.text] = 1
        else:    
            verb_frequencies[token.text] +=1                        
                
tags_pos_table.reset_index(drop=True, inplace=True)
print(tags_pos_table.shape)
tags_pos_table.head(20)


# In[142]:


# converting noun and verb dictionary to dataframe for sorting and plotting

list_noun_key = list(noun_frequencies.keys()) #converts dictionary keys to list of keys
list_noun_val = list(noun_frequencies.values()) # converts dictionary values to list of values
list_verb_key = list(verb_frequencies.keys())
list_verb_val = list(verb_frequencies.values())

# creating table for noun words frequency
noun_table = pd.DataFrame()
noun_table['noun']       = list_noun_key
noun_table['noun_count'] = list_noun_val
noun_table.sort_values(by='noun_count', ascending=False, inplace=True)
noun_table_10            = noun_table.head(10)

#creating table for verb word frequency
verb_table               = pd.DataFrame()
verb_table['verb']       = list_verb_key
verb_table['verb_count'] = list_verb_val
verb_table.sort_values(by='verb_count', ascending=False, inplace=True)
verb_table_10            = verb_table.head(10)


# In[143]:


noun_table_10


# In[144]:


fig = px.bar(noun_table_10, x='noun', y='noun_count',
             color='noun_count',
             height=400, title='Top 10 most common nouns')
fig.show()


# In[145]:


verb_table_10


# In[146]:


fig = px.bar(verb_table_10, x='verb', y='verb_count',
             color='verb_count',
             height=400, title='Top 10 most common verbs',
             color_continuous_scale=px.colors.diverging.Tealrose,                          
            )
fig.show()


# In[147]:


# from nltk
stop_words = set(stopwords.words('english'))
#stop_words.add('testingstopwordsaddition')
type(stop_words)
print(len(stop_words))
print(stop_words)


# In[148]:


#creating word_token from sentences
print("corpus length =",len(corpus))
word_tokens = word_tokenize(corpus)
print("lenght of tokens =",len(word_tokens))
print(word_tokens[0:100])


# In[149]:


#removing punctuation, stop words
filtered_tokens = []
for word in word_tokens:
    if word not in punctuation:
        if word not in stop_words:
            filtered_tokens.append(word)

print("length of tokens =",len(filtered_tokens))            
print(filtered_tokens[0:100])


# In[150]:


# most 10 common words
# creating dictionaries of keys as words and values as frequency of that word
word_frequencies = {} 
for word in filtered_tokens:
    if word not in word_frequencies.keys():
        word_frequencies[word] = 1
    else:
        word_frequencies[word] +=1
print(word_frequencies)


# In[151]:


max(word_frequencies.values())


# In[152]:


list_keys = list(word_frequencies.keys())
list_values = list(word_frequencies.values())
#list_keys
#list_values
#converting the dictionary to dataframe
#so that sorting and plotting will become easier
wordcount_table = pd.DataFrame()
wordcount_table['words'] = list_keys
wordcount_table['wordcount'] = list_values
word_count_t = wordcount_table.sort_values(by='wordcount', ascending=False).head(10)
word_count_t


# In[153]:


fig = px.bar(word_count_t, x='words', y='wordcount',
             color='wordcount',
             height=400, title='Top 10 most common words',
             color_continuous_scale=px.colors.sequential.Viridis             
            )
fig.show() 


# In[154]:


#bigrams = ngrams(corpus.split(), 2)
#for item in bigrams:
#    print(item)

#creating dicitionary of bigrams with their frequency
bigrams = ngrams(corpus.split(), 2)

bigram_dict = {}
for item in bigrams:
    #print(item)
    itemText = item[0]+' '+item[1]
    if itemText not in bigram_dict.keys():
        bigram_dict[itemText] = 1
    else:    
        bigram_dict[itemText] +=1 
    
bigram_dict


# In[155]:


#converting dictionary to dataframe for sorting and plotting
list_bigram_key = list(bigram_dict.keys()) #converts dictionary keys to list of keys
list_bigram_val = list(bigram_dict.values()) # converts dictionary values to list of values

# creating table for Bigrams
bigram_table = pd.DataFrame()
bigram_table['Bigrams']    = list_bigram_key
bigram_table['Frequency']  = list_bigram_val
bigram_table.sort_values(by='Frequency', ascending=False, inplace=True)
bigram_table_10            = bigram_table.head(10)
bigram_table_10


# In[156]:


# 10 most frequent bigrams 
bigram_table_10.plot('Bigrams','Frequency',kind='bar', width=0.6, color ='orange', title ='10 most frequent Bigrams')


# In[157]:


#trigrams = ngrams(corpus.split(), 3)
#for item in trigrams:
#    print(item)

#creating dicitionary of trigrams with their frequency
trigrams = ngrams(corpus.split(), 3)

trigram_dict = {}
for item in trigrams:
    #print(item)
    itemText = item[0]+' '+item[1]+' '+item[2]
    if itemText not in trigram_dict.keys():
        trigram_dict[itemText] = 1
    else:    
        trigram_dict[itemText] +=1 
    
trigram_dict


# In[158]:


#converting dictionary to dataframe for sorting and plotting
list_trigram_key = list(trigram_dict.keys()) #converts dictionary keys to list of keys
list_trigram_val = list(trigram_dict.values()) # converts dictionary values to list of values

# creating table for noun words frequency
trigram_table = pd.DataFrame()
trigram_table['Trigrams']    = list_trigram_key
trigram_table['Frequency']  = list_trigram_val
trigram_table.sort_values(by='Frequency', ascending=False, inplace=True)
trigram_table_10            = trigram_table.head(10)
trigram_table_10


# In[159]:


# 10 most frequent trigrams 
trigram_table_10.plot('Trigrams','Frequency', kind='bar', width=0.6, title ='10 most frequent Trigrams')


# 

# #### Using Wordcloud

# In[160]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                        background_color='black',
                        stopwords=stopwords,
                        max_words=200, 
                        random_state=42).generate(str(noun_table))

plt.figure(figsize=(15,10))
plt.axis("off")
plt.title("Words frequented in text", fontsize=15)
plt.imshow(wordcloud.recolor(colormap= 'viridis' , random_state=42), alpha=0.98)
plt.show()


# In[161]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                        background_color='black',
                        stopwords=stopwords,
                        max_words=200, 
                        random_state=42).generate(str(verb_table))

plt.figure(figsize=(15,10))
plt.axis("off")
plt.title("Words frequented in text", fontsize=15)
plt.imshow(wordcloud.recolor(colormap= 'viridis' , random_state=42), alpha=0.98)
plt.show()


# In[162]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                        background_color='black',
                        stopwords=stopwords,
                        max_words=200, 
                        random_state=42).generate(str(word_count_t))

plt.figure(figsize=(15,10))
plt.axis("off")
plt.title("Words frequented in text", fontsize=15)
plt.imshow(wordcloud.recolor(colormap= 'viridis' , random_state=42), alpha=0.98)
plt.show()


# In[163]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                        background_color='black',
                        stopwords=stopwords,
                        max_words=200, 
                        random_state=42).generate(str(bigram_table))

plt.figure(figsize=(15,10))
plt.axis("off")
plt.title("Words frequented in text", fontsize=15)
plt.imshow(wordcloud.recolor(colormap= 'viridis' , random_state=42), alpha=0.98)
plt.show()


# In[164]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                        background_color='black',
                        stopwords=stopwords,
                        max_words=200, 
                        random_state=42).generate(str(trigram_table))

plt.figure(figsize=(15,10))
plt.axis("off")
plt.title("Words frequented in text", fontsize=15)
plt.imshow(wordcloud.recolor(colormap= 'viridis' , random_state=42), alpha=0.98)
plt.show()


# # Automatic Summarizers

# ###  Extractive

# ##### Using Sumy

# In[165]:


language = "english"
sentence_count = 5
 
parser = PlaintextParser(corpus, Tokenizer(language))


# In[166]:


summarizer = LexRankSummarizer(Stemmer(language))
summarizer.stop_words = get_stop_words(language)
#Summarize the document with 2 sentences
summary = summarizer(parser.document, sentence_count) 
    
summary_pre_LexRankSummarizer = ' '.join([str(sentence) for sentence in summary])
summary_pre_LexRankSummarizer


# In[167]:


from sumy.summarizers.luhn import LuhnSummarizer
summarizer_1 = LuhnSummarizer(Stemmer(language))
summarizer_1.stop_words = get_stop_words(language)
summary_1 = summarizer_1(parser.document, sentence_count)

summary_pre_LuhnSummarizer = ' '.join([str(sentence) for sentence in summary_1])
summary_pre_LuhnSummarizer


# In[168]:


from sumy.summarizers.lsa import LsaSummarizer
summarizer_2 = LsaSummarizer(Stemmer(language))
summarizer_2.stop_words = get_stop_words(language)
summary_2 = summarizer_2(parser.document, sentence_count)

summary_pre_LsaSummarizer = ' '.join([str(sentence) for sentence in summary_2])
summary_pre_LsaSummarizer


# In[169]:


from sumy.summarizers.text_rank import TextRankSummarizer
summarizer_3 = TextRankSummarizer(Stemmer(language))
summarizer_3.stop_words = get_stop_words(language)
summary_3 = summarizer_3(parser.document, sentence_count)

summary_pre_TextRankSummarizer = ' '.join([str(sentence) for sentence in summary_3])
summary_pre_TextRankSummarizer


# In[170]:


from sumy.summarizers.edmundson import EdmundsonSummarizer
summarizer_4 = EdmundsonSummarizer(Stemmer(language))
summarizer_4.stop_words = get_stop_words(language)
summarizer_4.bonus_words = corpus.split()
summarizer_4.stigma_words = ['zdfgthdvndadv']
summarizer_4.null_words = stop_words
summary_4 = summarizer_4(parser.document, sentence_count)

summary_pre_EdmundsonSummarizer = ' '.join([str(sentence) for sentence in summary_4])
summary_pre_EdmundsonSummarizer


# In[171]:


from sumy.summarizers.reduction import ReductionSummarizer
summarizer_7 = ReductionSummarizer(Stemmer(language))
summarizer_7.stop_words = get_stop_words(language)
summary_7 = summarizer_7(parser.document, sentence_count)

summary_pre_ReductionSummarizer = ' '.join([str(sentence) for sentence in summary_7])
summary_pre_ReductionSummarizer


# ##### Using Gensim

# In[172]:


summary_using_gensim = summarize(corpus, word_count=100)
print(summary_using_gensim)


# ###   Abstractive

# In[173]:


model_t5 = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')  # model was trained on GPU, need to do when running on CPU


# In[174]:


preprocess_text = corpus.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text # need to add 'summarize:' to input text data
#print ("original text preprocessed: \n", preprocess_text)
print ("original text preprocessed: \n", len(preprocess_text))


# In[175]:


tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

# summmarize 
summary_ids = model_t5.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=100,
                                    max_length=400,
                                    early_stopping=True)

summary_pre_Transformer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",summary_pre_Transformer)


# # Model Building - Extractive Summerizer

# ### Word frequency -  Spacy

# In[176]:


nlp = spacy.load('en_core_web_sm')


# In[177]:


print("length of corpus =",len(corpus))
#passing entire corpus(string)
doc = nlp(corpus)
print("length of doc =",len(doc))
doc


# In[178]:


#print tokens from the doc
word_tokens = [tokens.text for tokens in doc]
print("length of tokens", len(word_tokens))
print(word_tokens)


# In[179]:


spacy_stopwords = list(STOP_WORDS)
print(spacy_stopwords)


# In[180]:


#removing punctuation and stopwords
#creates word frequency dictionary with words(keys) thier frequency(values)
word_frequencies = {} 
for word in word_tokens:
    if word not in punctuation:
        if word not in spacy_stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] +=1

print(word_frequencies)


# In[181]:


max_frequency = max(word_frequencies.values())
max_frequency


# In[182]:


#we will divide each value in word_frequency dict by maximum frequency(40)
#to get a normalized value for all words
#so, 40/40 = 1, is the maximum normalized frequency

for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_frequency
    
print(word_frequencies)


# In[183]:


sentence_tokens = [sent for sent in doc.sents]
print("length of sentence tokens =",len(sentence_tokens),"\n")
print(sentence_tokens)


# In[184]:


#to calculate sentence scores
#print(word_frequencies)
sentence_scores = {}

for sent in sentence_tokens:
        for word in sent: # for each word in sentence
            if word.text.lower() in word_frequencies.keys(): # if word exists in word frequency dict
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]#freq. of word from word_freq dict assig to sent_score dict
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]


# In[185]:


sentence_scores


# In[186]:


#30% of the sentences from the entire corpus will be selected
# ie 30% of 173 sentences = 51 sentence
select_length = int(len(sentence_tokens)*0.03)
select_length


# In[187]:


#will select sentences(select_length=5) with highest sentence score
summary = nlargest(select_length,sentence_scores,key=sentence_scores.get)
summary


# In[188]:


#converting summary to string
summary_spacy_wordfrequency = ' '.join([str(elem) for elem in summary])
print("Summary :\n",summary_spacy_wordfrequency)


# #### Sentiment Analysis Function

# In[189]:


#sentiment analysis
def summarysentiment(textsummary):
    list_assert_words = []
    afn = Afinn()
    scores = afn.score(textsummary)
    listAssertWords = afn.find_all(textsummary)
    #converting list to string
    strAssertWords = ', '
    strAssertWords = strAssertWords.join(listAssertWords)
    
    sentiment = ''
    if scores > 0:
        sentiment = 'Positive'
    elif scores < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return (sentiment,strAssertWords,scores)


# #### Sentiment Analysis

# In[190]:


#sentiment analysis
sentiment_summary, strAssertWords,scores = summarysentiment(summary_spacy_wordfrequency)

sumarry_sentiments                   = pd.DataFrame()
sumarry_sentiments['Summary']        = pd.DataFrame([[summary_spacy_wordfrequency]])
sumarry_sentiments['Assert words']   = strAssertWords
sumarry_sentiments['Score']          = pd.DataFrame([[scores]])
sumarry_sentiments['Review']         = pd.DataFrame([[sentiment_summary]])

sumarry_sentiments


# ###  TF-IDF  - NLTK

# In[191]:


#sentence tokenization from text 
sentences  = sent_tokenize(corpus)
print(sentences)
total_documents = len(sentences)
total_documents


# In[192]:


#calculate frequency of words in each sentence
from nltk.corpus import stopwords
def create_frequency_matrix(sentences):
    frequency_matrix = {}
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    for sent in sentences:
        #print(sent,"\n")
        freq_table = {}
        words = word_tokenize(sent)
        #print(words,"\n")
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stop_words:
                continue
            
            #print(word)
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1
                
        frequency_matrix[sent[:15]] = freq_table
        
    return frequency_matrix

#Create the Frequency matrix of the words in each sentence.
freq_matrix = create_frequency_matrix(sentences)
print(freq_matrix)


# In[193]:


#calculating term frequency
#Here, the document is a sentence, the term is a word in sentence
def create_tf_matrix(freq_matrix):
    tf_matrix = {}
    
    for sent,f_table in freq_matrix.items():
        #print(f_table)
        tf_table = {}
        count_words_sentence = len(f_table)
        #print(count_words_sentence)
        for word, count in f_table.items():
            tf_table[word] = count/count_words_sentence

        tf_matrix[sent] = tf_table
            
    return tf_matrix 

#calculate the term frequency and generate a matrix
tf_matrix = create_tf_matrix(freq_matrix)
print(tf_matrix)


# In[194]:


#calculating how many sentences contains a word
def create_documents_per_words(freq_matrix):
    word_per_doc_table = {}
    
    for sent, f_table in freq_matrix.items():
        #print(sent)
        #print(f_table)
        for word, count in f_table.items():
            if word in word_per_doc_table:
                #print(word)
                word_per_doc_table[word] +=1
            else:
                word_per_doc_table[word] = 1
                
    return word_per_doc_table

#creating table for documents per words
count_doc_per_words  = create_documents_per_words(freq_matrix)
print(count_doc_per_words)


# In[195]:


#calculating IDF for each word in a sentence
def create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}
    
    for sent, f_table in freq_matrix.items():
        idf_table = {}
        #print(sent)

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

# calculate IDF and generate a matrix
idf_matrix = create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
print(idf_matrix)


# In[196]:


#calculating TF-IDF (i.e TF*IDF)
def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

# Calculate TF-IDF and generate a matrix
tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)
print(tf_idf_matrix)


# In[197]:


#scoring sentences
def score_sentences(tf_idf_matrix) -> dict:
    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue

#score the sentences
sentence_scores = score_sentences(tf_idf_matrix)
print(sentence_scores)


# In[198]:


#calculating threshold = average sentece score
def find_average_score(sentenceValue) -> int:
    
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average

#Find the threshold
threshold = find_average_score(sentence_scores)
print(threshold)


# In[199]:


#select a sentence if the sentence score is greater than threshold
def generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        #print(sentence)
       # print(sentenceValue)
        #print(threshold)
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
        #if sentenceValue[sentence[:15]] >= (threshold):
            #print("inside if")
            #print(sentenceValue[sentence[:15]])
            #print(sentenceValue[sentence],"\n")
            summary += " " + sentence
            sentence_count += 1

    #print("==============",summary)
    return summary


# In[200]:


#Generate the summary
summary_nltk_tfidf = generate_summary(sentences, sentence_scores, 1.5 * threshold)
print("summary:\n",summary_nltk_tfidf)


# #### Sentiment Analysis

# In[201]:


#sentiment analysis
sentiment_summary_1, strAssertWords_1,scores_1 = summarysentiment(summary_nltk_tfidf)

sumarry_sentiments_1                   = pd.DataFrame()
sumarry_sentiments_1['Summary']        = pd.DataFrame([[summary_nltk_tfidf]])
sumarry_sentiments_1['Assert words']   = strAssertWords_1
sumarry_sentiments_1['Score']          = pd.DataFrame([[scores_1]])
sumarry_sentiments_1['Review']         = pd.DataFrame([[sentiment_summary_1]])

sumarry_sentiments_1


# ### BOW - CountVectorizer - Sklearn

# In[202]:


print(len(corpus))


# In[203]:


sentences_bow  = sent_tokenize(corpus)
print(sentences_bow[:5])


# In[204]:


# Use TextBlob
def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words


# In[205]:


#removes stopwords, stems the words and gives vector representation of the words of each sentence
vec_count = CountVectorizer(analyzer='word', stop_words='english', tokenizer=textblob_tokenizer)
bow = vec_count.fit_transform(sentences_bow)
bow.toarray()


# In[206]:


df_bow = pd.DataFrame(bow.toarray(), columns= vec_count.get_feature_names())
df_bow


# In[207]:


#adding sentences in dataframe
df_bow['sentence'] = sentences_bow
df_bow.head(5)


# In[208]:


df_bow['sentence score'] = df_bow.sum(axis=1)
df_bow['sentence score']


# In[209]:


#sorting on the basis of sentence score
df_bow.sort_values(by='sentence score', ascending=False, inplace=True)
df_bow[['sentence','sentence score']]


# In[210]:


summary_length  = 5
df_bow_topscore = df_bow[['sentence','sentence score']].head(summary_length)
print(df_bow_topscore.shape)
df_bow_topscore


# In[211]:


df_bow_topscore = df_bow_topscore.reset_index(drop=True)
print(len(df_bow_topscore))
df_bow_topscore


# In[212]:


summary_bow = ''
for i in range(len(df_bow_topscore)):
    summary_bow += df_bow_topscore['sentence'][i]
    
summary_bow


# #### Sentiment Analysis

# In[213]:


#sentiment analysis
sentiment_summary_1, strAssertWords_1,scores_1 = summarysentiment(summary_bow)

sumarry_sentiments_1                   = pd.DataFrame()
sumarry_sentiments_1['Summary']        = pd.DataFrame([[summary_bow]])
sumarry_sentiments_1['Assert words']   = strAssertWords_1
sumarry_sentiments_1['Score']          = pd.DataFrame([[scores_1]])
sumarry_sentiments_1['Review']         = pd.DataFrame([[sentiment_summary_1]])

sumarry_sentiments_1


# ### TfidfVectorizer - Sklearn

# In[214]:


print(len(corpus))


# In[215]:


sentences  = sent_tokenize(corpus)
print(sentences)
total_documents = len(sentences)
total_documents


# In[216]:


# Using NLTK's PorterStemmer
ps = PorterStemmer()
def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [ps.stem(word) for word in words]
    return words


# In[217]:


#removes stopwords, stems the words and gives vector representation of the words of each sentence
vectorizer = TfidfVectorizer(analyzer='word', stop_words = 'english', tokenizer = stemming_tokenizer )
tfidf = vectorizer.fit_transform(sentences)
tfidf.toarray()


# In[218]:


df = pd.DataFrame(tfidf.toarray(), columns= vectorizer.get_feature_names())
df


# In[219]:


#adding sentences in dataframe
df['sentence'] = sentences
df.head(5)


# In[220]:


df['sentence score'] = df.sum(axis=1)
df['sentence score']


# In[221]:


#sorting on the basis of sentence score
df.sort_values(by='sentence score', ascending=False, inplace=True)
df[['sentence','sentence score']]


# In[222]:


summary_length  = 5
df_tfidf_topscore = df[['sentence','sentence score']].head(summary_length)
print(df_tfidf_topscore.shape)
df_tfidf_topscore


# In[223]:


df_tfidf_topscore = df_tfidf_topscore.reset_index(drop=True)
print(len(df_tfidf_topscore))
df_tfidf_topscore


# In[224]:


summary_tfidf = ''
for i in range(len(df_tfidf_topscore)):
    summary_tfidf += df_tfidf_topscore['sentence'][i]
    
summary_tfidf


# #### Sentiment Analysis

# In[225]:


#sentiment analysis
sentiment_summary_1, strAssertWords_1,scores_1 = summarysentiment(summary_tfidf)

sumarry_sentiments_1                   = pd.DataFrame()
sumarry_sentiments_1['Summary']        = pd.DataFrame([[summary_tfidf]])
sumarry_sentiments_1['Assert words']   = strAssertWords_1
sumarry_sentiments_1['Score']          = pd.DataFrame([[scores_1]])
sumarry_sentiments_1['Review']         = pd.DataFrame([[sentiment_summary_1]])

sumarry_sentiments_1


# # Model Evaluation - Rouge Score

# ### Word Frequency Model (Spacy) -  Rouge Score

# In[226]:


#comparing wordfrequency model with Sumy - ReductionSummarizer
r = Rouge()
r.get_scores(summary_spacy_wordfrequency, summary_pre_ReductionSummarizer)


# In[227]:


#comparing wordfrequency model with Sumy - EdmundsonSummarizer
r = Rouge()
r.get_scores(summary_spacy_wordfrequency, summary_pre_EdmundsonSummarizer)


# In[228]:


#comparing wordfrequency model with Sumy - LexRankSummarizer
r = Rouge()
r.get_scores(summary_spacy_wordfrequency, summary_pre_LexRankSummarizer)


# In[229]:


#comparing wordfrequency model with Sumy - LuhnSummarizer
r = Rouge()
r.get_scores(summary_spacy_wordfrequency, summary_pre_LuhnSummarizer)


# In[230]:


#comparing wordfrequency model with Sumy - LsaSummarizer
r = Rouge()
r.get_scores(summary_spacy_wordfrequency, summary_pre_LsaSummarizer)


# In[231]:


#comparing wordfrequency model with Sumy - TextRankSummarizer
r = Rouge()
r.get_scores(summary_spacy_wordfrequency, summary_pre_TextRankSummarizer)


# In[232]:


#comparing wordfrequency model with Gensim
r = Rouge()
r.get_scores(summary_spacy_wordfrequency, summary_using_gensim)


# In[233]:


#comparing wordfrequency model with T5 Transformer
r = Rouge()
r.get_scores(summary_spacy_wordfrequency, summary_pre_Transformer)


# #####  The best accuracy we are getting in the Word Frequency Model is 98% after we compared it to the summaries generated by the Pre-Trained Models using Rouge score.

# ### TF-IDF Model (NLTK) - Rouge Score

# In[234]:


#comparing tf-idf model with Sumy - ReductionSummarizer
r = Rouge()
r.get_scores(summary_nltk_tfidf, summary_pre_ReductionSummarizer)


# In[235]:


#comparing tf-idf model with Sumy - EdmundsonSummarizer
r = Rouge()
r.get_scores(summary_nltk_tfidf, summary_pre_EdmundsonSummarizer)


# In[236]:


#comparing tf-idf model with Sumy - LexRankSummarizer
r = Rouge()
r.get_scores(summary_nltk_tfidf, summary_pre_LexRankSummarizer)


# In[237]:


#comparing tf-idf model with Sumy - LuhnSummarizer
r = Rouge()
r.get_scores(summary_nltk_tfidf, summary_pre_LuhnSummarizer)


# In[238]:


#comparing tf-idf model with Sumy - LsaSummarizer
r = Rouge()
r.get_scores(summary_nltk_tfidf, summary_pre_LsaSummarizer)


# In[239]:


#comparing tf-idf model with Sumy - TextRankSummarizer
r = Rouge()
r.get_scores(summary_nltk_tfidf, summary_pre_TextRankSummarizer)


# In[240]:


#comparing tf-idf model with Gensim
r = Rouge()
r.get_scores(summary_nltk_tfidf, summary_using_gensim)


# In[241]:


#comparing tf-idf model with T5 Transformer
r = Rouge()
r.get_scores(summary_nltk_tfidf, summary_pre_Transformer)


# #####  The best accuracy we are getting in the TF-IDF Model is 25% after we compared it to the summaries generated by the Pre-Trained Models using Rouge score.

# ### BOW - CountVectorizer (Sklearn) - Rouge Score

# In[242]:


#comparing BOW model with Sumy - ReductionSummarizer
r = Rouge()
r.get_scores(summary_bow, summary_pre_ReductionSummarizer)


# In[243]:


#comparing BOW model with Sumy - EdmundsonSummarizer
r = Rouge()
r.get_scores(summary_bow, summary_pre_EdmundsonSummarizer)


# In[244]:


#comparing BOW model with Sumy - LexRankSummarizer
r = Rouge()
r.get_scores(summary_bow, summary_pre_LexRankSummarizer)


# In[245]:


#comparing BOW model with Sumy - LuhnSummarizer
r = Rouge()
r.get_scores(summary_bow, summary_pre_LuhnSummarizer)


# In[246]:


#comparing BOW model with Sumy - LsaSummarizer
r = Rouge()
r.get_scores(summary_bow, summary_pre_LsaSummarizer)


# In[247]:


#comparing BOW model with Sumy - TextRankSummarizer
r = Rouge()
r.get_scores(summary_bow, summary_pre_TextRankSummarizer)


# In[248]:


#comparing BOW model with T5 Transformer
r = Rouge()
r.get_scores(summary_bow, summary_pre_Transformer)


# #####  The best accuracy we are getting in the BOW - CountVectorizer Model is 83% after we compared it to the summaries generated by the Pre-Trained Models using Rouge score.

# ### TfidfVectorizer (Sklearn) - Rouge Score

# In[249]:


#comparing TfidfVectorizer model with Sumy - ReductionSummarizer
r = Rouge()
r.get_scores(summary_tfidf, summary_pre_ReductionSummarizer)


# In[250]:


#comparing TfidfVectorizer model with Sumy - EdmundsonSummarizer
r = Rouge()
r.get_scores(summary_tfidf, summary_pre_EdmundsonSummarizer)


# In[251]:


#comparing TfidfVectorizer model with Sumy - LexRankSummarizer
r = Rouge()
r.get_scores(summary_tfidf, summary_pre_LexRankSummarizer)


# In[252]:


#comparing TfidfVectorizer model with Sumy - LuhnSummarizer
r = Rouge()
r.get_scores(summary_tfidf, summary_pre_LuhnSummarizer)


# In[253]:


#comparing TfidfVectorizer model with Sumy - LsaSummarizer
r = Rouge()
r.get_scores(summary_tfidf, summary_pre_LsaSummarizer)


# In[254]:


#comparing TfidfVectorizer model with Sumy - TextRankSummarizer
r = Rouge()
r.get_scores(summary_tfidf, summary_pre_TextRankSummarizer)


# #####  The best accuracy we are getting in the TF-IDF Vectorizer Model is 74% after we compared it to the summaries generated by the Pre-Trained Models using Rouge score.

# # Model Deployment

# In[255]:


import joblib 
# joblib.dump(sentiment_classifier, 'sentiment_model_pipeline.pkl')


# In[256]:


# from pickle import dump
# dump(log_model,open('claimants_model.pkl','wb'))
# from pickle import load
# loaded_log_model = load(open('claimants_model.pkl','rb'))
# loaded_log_model.score(X_test,y_test)


# In[ ]:




