from transformers import pipeline
import pandas as pd
import numpy as np

def accuracy(ground_truth,predicted):
  cnt=0
  for i,j in zip(ground_truth,predicted):
    if i.upper()==j:
      cnt+=1
    else:
      pass
  return cnt/len(ground_truth)

classifier = pipeline('sentiment-analysis')
data = pd.read_csv('tweets-sentiment-synth.csv')
texts = [text for text in data.tweet]
outputs = classifier(texts)
predictions = [output['label'] for output in outputs]

print('model accuracy is:',accuracy(data.sentiment,predictions))
