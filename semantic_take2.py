import pandas as pd
import os
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
# import networkx as nx
import matplotlib.pyplot as plt
# import itertools
from sklearn.manifold import TSNE

#tokenized_docs = []

def g():
	nltk.download('punkt')
	nltk.download('stopwords')
	nltk.download('wordnet')

	# Initialize NLP tools
	stop_words = set(stopwords.words('english'))
	lemmatizer = WordNetLemmatizer()

	def preprocess_text(text):
		tokens = word_tokenize(text)
		tokens = [w.lower() for w in tokens if w.isalpha()]
		tokens = [w for w in tokens if not w in stop_words]
		tokens = [lemmatizer.lemmatize(w) for w in tokens]
		#I guess this didn't stem them? I thought it did.
		return tokens


	mydir = 'C:\\Users\\me\\Documents\\biomotivate\\biomo-need-sol\\data'
	dfs = []

	for filename in os.listdir(mydir):
		if filename.endswith('regexed.csv'):
			filepath = os.path.join(mydir, filename)
			dfs.append(pd.read_csv(filepath))

	raw = pd.concat(dfs).drop_duplicates()
	#raw.to_csv('mydocs.csv', index=False, encoding='utf-8-sig')
	tokenized_docs = [preprocess_text(x) for x in raw['body']]
#print(len(tokenized_docs)) #4k-ish
#where tokenized_docs[i] is a list of words

#print("Creating model...")
#model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=10, workers=4)
#model.save('w2vmodelv1.model')
print("Loading model...")
model = Word2Vec.load("C:\\Users\\me\\Documents\\biomotivate\\biomo-need-sol\\w2vmodelv1.model")

mytestwords = ['solution', 'solve', 'try', 'help', 'fix', 
'suggestion', 'suggest', 'advise', 'advice', 'alanon', 
'resource', 'tool', 'guide', 'guidance', 'connect', 
'support', 'supportive', 'find', 'found', 'worked', 'ask']

def onegraph(ax, subL, title):
	subL = sorted(subL, key=lambda x: x[1], reverse=False)
	words = [x[0] for x in subL]
	vals = [x[1] for x in subL]
	ax.barh(words, vals)
	# ax.set_xlabel('Cosine Similarity')
	# ax.set_ylabel('Word')
	ax.set_xlim(left=0.9, right=1)
	ax.set_title(title, pad=-20)

def mygraph(wordlist, topn, ncols=3):
	tmp = []

	for word in wordlist:
		try:
			#print("similar to", word,":")
			tmp.append(model.wv.most_similar(word))
		except Exception as e:
			print(e)
	#word2vec similarity is calculated using cosine similarity equation. i.e. literally the mathematical distace between two vectors.
	#tmp3 = sorted(tmp, key = lambda x: x[1], reverse = True)
	#actually its already sorted so idc
	nrows = len(wordlist)//ncols + 1
	#nrows=3 #wtf
	fig, axs = plt.subplots(nrows, ncols, figsize = (12, 6*nrows))

	for i, L in enumerate(tmp):
		r = i//ncols
		c = i % ncols
		onegraph(axs[r, c], L[0:topn], wordlist[i])

	for j in range(ncols - nrows % ncols):
		fig.delaxes(axs[-1, -(j+1)])

	#plt.subplots_adjust(hspace=0.516, wspace=0.215, top=0.9, right=0.98, bottom=0.05, left=0.09)

	plt.tight_layout()
	plt.show()

#mygraph(mytestwords, 5, 7)

def f():
	print("Making TSNE...")
	vocab = list(model.wv.key_to_index)
	X = model.wv[vocab]
	tsne = TSNE(n_components = 2)
	X_tsne = tsne.fit_transform(X)
	tsne_df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
	#the words are indeces of this df
	print(tsne_df.size)

	print("Making graph...")
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(tsne_df['x'], tsne_df['y'])
	for word, pos in tsne_df.iterrows():
		ax.annotate(word, pos)
	plt.show()
f()