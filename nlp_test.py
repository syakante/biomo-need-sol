import string
from collections import Counter
import pandas as pd

import nltk
from nltk.downloader import Downloader
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
#from gensim.models import Word2Vec

#https://stackoverflow.com/questions/23704510/how-do-i-test-whether-an-nltk-resource-is-already-installed-on-the-machine-runni
def download_nltk_data(list_of_resources) -> None:
	for resource in list_of_resources:
		nltk.download(info_or_id = resource,
					  #download_dir = download_dir,
					  quiet = True)
	return

download_nltk_data(
	list_of_resources=['vader_lexicon',
					   'punkt',
					   'stopwords',
					   'wordnet',
					   'averaged_perceptron_tagger',
					   'universal_tagset'])

mystopwords = set(stopwords.words('english'))
mylemmatizer = WordNetLemmatizer()
mysentiment = SentimentIntensityAnalyzer()

def preprocess_text(text:str):
	#remove punctuation, tokenize, remove stopwords, lemmatize
	text = text.translate(str.maketrans('', '', string.punctuation))
	tokens = word_tokenize(text.lower())
	tokens = [token for token in tokens if token not in mystopwords]
	tokens = [mylemmatizer.lemmatize(token) for token in tokens]
	#return(' '.join(tokens))
	return(tokens)

def num_unique_words(text:str):
	return(len(set(text.split(' '))))

def get_sentiment(text:str):
	scores = mysentiment.polarity_scores(text)
	return(scores) #dictionary of neg, neu, pos, compound

def get_pos_counts(tokenized_text):
	#returns a dict
	pos_counts = Counter([j for i, j in pos_tag(tokenized_text, tagset='universal')])
	#for now, just want adj, verbs, adverbs
	#ADJ, VERB, ADV
	my_pos_count = {k:pos_counts.get(k, 0) for k in ['ADJ', 'VERB', 'ADV']}
	return my_pos_count

df = pd.read_csv('comments.csv', encoding='utf-8-sig')
#v definitely can make it more efficient but im lazy rn
print("read csv.")

df = df[df['body'].map(num_unique_words) > 20] #idk
#this makes the tables unaligned at some point and idk why :|
print("now have", df.shape[0], "rows")

tokenized_col = [preprocess_text(x) for x in df['body']]
print("processed text.")
df['unique_words'] = [len(set(x)) for x in tokenized_col]
print("counted words.")
sent_df = pd.DataFrame([get_sentiment(x) for x in df['body']])
print("got sentiments.")
df = pd.concat([df, sent_df], axis=1)
pos_df = pd.DataFrame([get_pos_counts(x) for x in tokenized_col])
print("got POS.")
df = pd.concat([df, pos_df], axis=1)
print("ok")
df.to_csv("ling_feats.csv", index=False, encoding='utf-8-sig')
print("wrote.")