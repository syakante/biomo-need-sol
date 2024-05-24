import pandas as pd
import os
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import yake
from gensim.models import Word2Vec
#import networkx as nx
#import matplotlib.pyplot as plt
import itertools
import sklearn.manifold.TSNE

#tokenized_docs = []

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
	return tokens


mydir = 'C:\\Users\\me\\Documents\\biomotivate\\biomo-need-sol\\data'
dfs = []

for filename in os.listdir(mydir):
	if filename.endswith('regexed.csv'):
		filepath = os.path.join(mydir, filename)
		dfs.append(pd.read_csv(filepath))

raw = pd.concat(dfs).drop_duplicates()
#oh lmao i forgor to preprocess it
tokenized_docs = [preprocess_text(x) for x in raw['body']]
print(len(tokenized_docs))
# Train Word2Vec model
#as_sent = 

#model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)
print("Loading model...")
model = Word2Vec.load("C:\\Users\\me\\Documents\\biomotivate\\word2vec_model.model")

print("Making YAKE...")
# Adjust YAKE to extract unigrams
kw_extractor = yake.KeywordExtractor(n=1, top=20)  # Extract top 20 unigrams

# Keyword extraction (adjust according to your needs)
keywords = set()
for doc in tokenized_docs:
	doc_text = ' '.join(doc)
	extracted_keywords = kw_extractor.extract_keywords(doc_text)
	for kw, _ in extracted_keywords:
		keywords.add(kw.split()[0])  # Assuming unigrams, but safe-guarding split
print(len(keywords))
print("Making graph...")
# Create a semantic network

#tsne = sklearn.manifold.TSNE(n_components = 0, random_state = 0)
#all_vector_matrix = 


#v it makes a yuuuuuuge graph that takes like an hour to make x_x
def tmp():
	G = nx.Graph()
	for word in keywords:
		if word in model.wv:
			G.add_node(word)
			for other_word in keywords:
				if other_word in model.wv and other_word != word:
					similarity = model.wv.similarity(word, other_word)
					if similarity > 0.5:  # Example threshold
						G.add_edge(word, other_word, weight=similarity)

	if len(G.nodes) == 0:
		print("The graph G has no nodes.")
	else:
		print(f"The graph G has {len(G.nodes)} nodes.")

	if len(G.edges) == 0:
		print("The graph G has no edges.")
	else:
		print(f"The graph G has {len(G.edges)} edges.")

	single_word_keywords = [kw[0] for kw in extracted_keywords]

	# Now, check if these single words are in the model's vocabulary
	for keyword in single_word_keywords[:10]:  # Check the first 10 keywords
		if keyword not in model.wv:
			print(f"'{keyword}' is NOT in the model's vocabulary.")

	# Assuming G is your original graph with 465 nodes and 48700 edges

	# Step 1: Define a weight threshold for edge filtering
	weight_threshold = 0.7  # Adjust this threshold based on your edge weights

	print("More graph stuff here")
	# Filter edges by weight
	edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0) > weight_threshold]
	G_filtered_by_weight = nx.Graph()
	G_filtered_by_weight.add_edges_from(edges_to_keep)

	# Add nodes to ensure the graph's integrity
	for node in G.nodes():
		if not G_filtered_by_weight.has_node(node):
			G_filtered_by_weight.add_node(node)

	# Step 2: Optionally, further filter nodes based on degree or other criteria
	# For example, removing nodes with degree less than a threshold
	degree_threshold = 130  # Adjust this threshold based on your criteria
	nodes_to_remove = [node for node, degree in dict(G_filtered_by_weight.degree()).items() if degree < degree_threshold]
	G_final_filtered = G_filtered_by_weight.copy()
	G_final_filtered.remove_nodes_from(nodes_to_remove)

	# Visualization of the final filtered graph with adjusted labels
	pos = nx.spring_layout(G_final_filtered)  # Use the same position layout as before

	# Calculate node degrees
	degrees = dict(G_final_filtered.degree())

	# Scale node sizes by degree (you might need to adjust the scaling factor)
	node_sizes = [degrees[node] * 10 for node in G_final_filtered.nodes()]  # Adjust scaling factor as needed

	# Visualization of the final filtered graph
	pos = nx.spring_layout(G_final_filtered)  # Adjust layout as necessary
	plt.figure(figsize=(20, 12))
	nx.draw_networkx(G_final_filtered, pos, with_labels=False, node_size=node_sizes, edge_color='lightgray', alpha=0.8)  # Adjust visual parameters as needed
	# Draw labels with adjusted positions and make them black
	nx.draw_networkx_labels(G_final_filtered,pos, font_color='black')
	plt.show()
