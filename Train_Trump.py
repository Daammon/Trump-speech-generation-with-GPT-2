from modeling_gpt2 import*
import torch
import pandas as pd
import numpy as np
from tokenization_gpt2 import*
import random
import torch.optim as optim
from bs4 import BeautifulSoup
import urllib3
import requests
import json
from torch.utils.data import Dataset, DataLoader

import logging
logging.basicConfig(level=logging.INFO)


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.
    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(length)
        while True:
            index = random.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]

# df = pd.read_csv('MrTrumpSpeeches.csv', sep='\~', quoting=3, engine='python')
#
# df = df['subtitles']
#
# i=0
#
# for speech in df:
#     new_text = re.sub('\[Music\]', '', speech)
#     new_text = re.sub('\[Applause\]', '', new_text)
#     new_text = re.sub('   ', ' ', new_text)
#     df[i]=new_text
#     i = i+1
#
# df.to_csv('Trump_Clean.csv')
# # print(df.head())
#
# out = ' '.join(df)
# print(out)

# Get speech urls
urls = []
for page in range(80):
  url = "https://factba.se/json/json-transcript.php?p="+str(page)
  response = requests.get(url)
  soup = BeautifulSoup(response.content, "html.parser")
  dictionary=json.loads(str(soup))
  d_data = dictionary['data']
  print(page)
  for speech in d_data:
    urls.append("https://factba.se/transcript/"+speech['slug'])

# Get all speeches
speeches = []
for url in urls:
  response = requests.get(url)
  soup = BeautifulSoup(response.content, "html.parser")
  print(url)
  phrases = soup.find_all('div', class_ = 'transcript-text-block')
  speech =''
  for phrase in phrases:
    speech = speech +' '+ phrase.a.text
  speeches.append(speech)

batch_size=1
df = pd.Series(speeches)
df.to_csv('factba_Trump.csv')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

token_chunks = []

for speech in speeches:
    tokens = np.stack(tokenizer.encode(speech))
    token_chunks.append(tokens)
data_sampler = Sampler(token_chunks)


# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')
optimizer = optim.Adam(model.parameters(), lr=0.0001)
model.to('cuda')


running_loss = 0.0
for mini_batch in range(200):
    # zero the parameter gradients
    optimizer.zero_grad()
    batch = [data_sampler.sample(512) for _ in range(batch_size)]
    batch = np.stack(batch, axis=0)
    batch = torch.from_numpy(batch).to('cuda')
    labels = batch.clone()
    labels[0][:-1] = labels[0][1:]
    labels[0][-1] = -1

    loss = model(input_ids=batch, lm_labels=labels)
    loss.backward()
    optimizer.step()
    # print statistics
    running_loss += loss.item()
    if mini_batch % 2 == 1:  # print every 2 mini-batches
        print('[%3d] loss: %.5f' %
              (mini_batch + 1, running_loss / 2))
        running_loss = 0.0
torch.save(model.state_dict(), 'gpt2_Trump1000.pt')


