import pickle as pkl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
from textblob import Word
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# import tensorflow
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split 
import wordcloud
import matplotlib.pyplot as plt

#Loading the labelled dataset of Tweets
with open("abb_labelled_df.pkl",  "rb") as fr:
    df = pkl.load(fr)

# print(df.head())
print(df.Label.value_counts())
# print(df[df["Label"] == "Negative"])


#Pre-Processing the text 
def cleaning(df, stop_words):
    df['Tweets'] = df['Tweets'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
    # Replacing the digits/numbers
    df['Tweets'] = df['Tweets'].str.replace('d', '')
    # Removing stop words
    df['Tweets'] = df['Tweets'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))
    # Lemmatization
    df['Tweets'] = df['Tweets'].apply(lambda x: ' '.join([Word(x).lemmatize() for x in x.split()]))
    return df
stop_words = stopwords.words('english')
data_cleaned = cleaning(df, stop_words)

common_words=''
for i in df.Tweets:
    i = str(i)
    tokens = i.split()
    common_words += " ".join(tokens)+" "
wordcloud = wordcloud.WordCloud().generate(common_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# plt.show()

# Encoded the target column
lb=LabelEncoder()
data_cleaned['Label'] = lb.fit_transform(data_cleaned['Label'])

#Generating Embeddings using tokenizer
tokenizer = Tokenizer(num_words=500, split=' ') 
tokenizer.fit_on_texts(data_cleaned['Tweets'].values)
X = tokenizer.texts_to_sequences(data_cleaned['Tweets'].values)
X = pad_sequences(X)

#Model Building
model = Sequential()
model.add(Embedding(500, 120, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(704, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(352, activation='LeakyReLU'))
model.add(Dense(3, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())

#Splitting the data into training and testing
y=pd.get_dummies(data_cleaned['Label'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

#Model Training
batch_size=32
model.fit(X_train, y_train, epochs = 10, batch_size=batch_size, verbose = 'auto')

# model.fit(X_train, y_train, epochs = 20, batch_size=32, verbose =1)

# Model Testing
print(r"NN test score on 20% split: ")
model.evaluate(X_test,y_test)