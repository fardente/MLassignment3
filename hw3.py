import os
import numpy as np
from glove import Corpus, Glove
from keras.layers import Convolution1D, MaxPooling1D, Embedding
from keras.layers import Convolution2D, MaxPooling2D, Dense, Input, Flatten
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = load_files('txt_sentoken')
#1)
# I played around with the parameters and these gave the best result, keeping stopwords actually seems to help!
count_vect = CountVectorizer(input='content', ngram_range=(1,2), min_df=2, max_df=200)
matrix = count_vect.fit_transform(data.data)

lm = LogisticRegression().fit(matrix, data.target)

acc = cross_val_score(lm, matrix, data.target, cv=10)
print("1) Acc. mean: " + str(acc.mean()))

#2)
idf_vect = TfidfVectorizer(input='content', ngram_range=(1,2),max_features=20000)
idf_matrix = idf_vect.fit_transform(data.data)
x = idf_matrix
y = data.target
folds = StratifiedKFold(y, 10)
x.shape
y.shape
x =x.toarray()

def train_test(model, x, y, folds):
    """ This function trains and tests a Keras model with k-fold cv.
        'folds' is the array returned by sklearn *KFold splits.
    """
    acc_sum = 0
    for trn_i, test_i in folds:
        model.fit(x[trn_i], y[trn_i], nb_epoch=1)
        _ , acc = model.evaluate(x[test_i], y[test_i])
        acc_sum += acc
    return acc_sum/len(folds)


def mlp_model(nhidden):
    """ This function creates two-layer networks
    """
    m = Sequential()
    m.add(Dense(input_dim=20000, output_dim=nhidden, activation='relu'))
    m.add(Dense(output_dim=1, activation='sigmoid'))
    m.compile(loss='binary_crossentropy', optimizer='sgd',
              metrics=['accuracy'])
    return m

mlp1 = mlp_model(1)
mlp1_accuracy = train_test(mlp1, x, y, folds)

mlp10 = mlp_model(10)
mlp10_accuracy = train_test(mlp10, x, y, folds)

mlp100 = mlp_model(100)
mlp100_accuracy = train_test(mlp100, x, y, folds)

mlp1000 = mlp_model(1000)
mlp1000_accuracy = train_test(mlp1000, x, y, folds)

print((mlp1_accuracy, mlp10_accuracy, mlp100_accuracy, mlp1000_accuracy))

#3CNN
#Glove Vectors from reviews
c = [review.split() for review in data.data]

corpus = Corpus()
corpus.fit(c, window=10)

glv = Glove(no_components=100, learning_rate=0.05)
glv.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)

glv.add_dictionary(corpus.dictionary)

embeddings_index = glv.dictionary

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = 'txt_sentoken/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            f = open(fpath, encoding="UTF-8")
            texts.append(f.read())
            f.close()
            labels.append(label_id)

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model 1D')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Convolution1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Convolution1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Convolution1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train, validation_data=(x_val, y_val),
nb_epoch=2, batch_size=128)