from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import pandas as pd

plt.rcParams.update({'font.size': 18, 'figure.figsize': (8,8), 'axes.axisbelow':True})

embeddings = torch.load('test_emb.pt')
test = pd.read_csv('test.csv')
test_target = np.array(test['target'])

subsample = random.sample(list(range(98000)),5000)
embeddings = embeddings[subsample,:]
subsample = np.array(subsample)
test_target = test_target[subsample]

tsne = TSNE(n_components=2)
visual_data = tsne.fit_transform(embeddings.cpu())
visual_data = visual_data/np.max(np.abs(visual_data))

print(visual_data.shape)
french = [a and b for a, b in zip(subsample>=40000,subsample<80000)]

english_neg = [a and b for a, b in zip(subsample<40000,test_target==0)]
english_pos = [a and b for a, b in zip(subsample<40000,test_target==1)]
french_neg = [a and b for a, b in zip(french,test_target==0)]
french_pos = [a and b for a, b in zip(french,test_target==1)]
russian_neg = [a and b for a, b in zip(subsample>=80000,test_target==0)]
russian_pos = [a and b for a, b in zip(subsample>=80000,test_target==1)]


plt.scatter(visual_data[english_neg][:,0],visual_data[english_neg][:,1],marker='x',color='#d7191c')
plt.scatter(visual_data[english_pos][:,0],visual_data[english_pos][:,1],marker='.',color='#d7191c')
plt.scatter(visual_data[french_neg][:,0],visual_data[french_neg][:,1],marker='x',color='#fdae61')
plt.scatter(visual_data[french_pos][:,0],visual_data[french_pos][:,1],marker='.',color='#fdae61')
plt.scatter(visual_data[russian_neg][:,0],visual_data[russian_neg][:,1],marker='x',color='#2b83ba')
plt.scatter(visual_data[russian_pos][:,0],visual_data[russian_pos][:,1],marker='.',color='#2b83ba')
plt.savefig('text.png')
plt.savefig('text.pdf')