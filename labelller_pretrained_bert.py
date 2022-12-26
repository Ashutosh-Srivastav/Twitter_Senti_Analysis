import pickle as pkl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

with open("abb_raw_UL_df.pkl",  "rb") as fr:
    df = pkl.load(fr)
# print(df.head())

def tweet_preprocessor(tweet):
    # precprcess tweet
    tweet_words = []

    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)
    return tweet_proc

# df["pre_proc_tweet"] = df.Tweets.apply(lambda x: tweet_preprocessor(x))

# print(df.head(5))

# load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

def sent_analyze(tweet):
    # sentiment analysis
    tweet = tweet_preprocessor(tweet)
    encoded_tweet = tokenizer(tweet, return_tensors='pt')
    # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    

    a = (np.argmax(scores))
    for i in range(len(scores)):
    
        l = labels[i]
        s = scores[i]
        print(l,s)
    print(labels[a])

    return labels[a]


df["Label"] = df.Tweets.apply(lambda x: sent_analyze(x))

with open("abb_labelled_df.pkl","wb") as f:
    pkl.dump(df, f)