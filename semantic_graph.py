from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

model = Word2Vec.load("C:\\Users\\me\\Documents\\biomotivate\\biomo-need-sol\\w2vmodelv1.model")
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