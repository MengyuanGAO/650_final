import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm
import time
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate, Lambda
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import load_model


def data_preparation(predict=False, sub_train=True):
    start_time = time.time()
    train_df = pd.read_csv("./train.csv")
    if sub_train:
        train_df = train_df.sample(frac=0.3)
    print("Train shape : ",train_df.shape)
    if predict:
        test_df = pd.read_csv("./test.csv")
        print("Test shape : ",test_df.shape)
        test_X = test_df["question_text"].fillna("_##_").values    
    ## split to train and val
    train_df, val_df = train_test_split(train_df, test_size=0.08, random_state=2018)
    
    ## fill up the missing values
    train_X = train_df["question_text"].values
    val_X = val_df["question_text"].values

    
    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features, filters='')
    print('fitting text to tokenizer..')
    check_point1 = time.time()
    tokenizer.fit_on_texts(list(train_X))
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    check_point2 = time.time()
    print('fitting took {:.2f} seconds to finish'.format(check_point2 - check_point1))
    save_text_tokenizer(tokenizer, "tokenizer")
    
    print('transforming text to sequence of word indices..')
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    check_point3 = time.time()
    print('transforming took {:.2f} seconds to finish'.format(check_point3 - check_point2))
    if predict:
        test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences 
    print('padding sentence to the same length..')
    train_X = pad_sequences(train_X, maxlen=maxlen)
    val_X = pad_sequences(val_X, maxlen=maxlen)
    check_point4 = time.time()
    print('padding took {:.2f} seconds to finish'.format(check_point4 - check_point3))
    
    if predict:
        test_X = pad_sequences(test_X, maxlen=maxlen)
        
    print('it took {:.2f} seconds to finish data prepartation'.format(time.time() - start_time))

    ## Get the target values
    train_y = train_df['target'].values
    val_y = val_df['target'].values  
    
#     trn_idx = np.random.permutation(len(train_X))
#     val_idx = np.random.permutation(len(val_X))

#     train_X = train_X[trn_idx]
#     val_X = val_X[val_idx]
#     train_y = train_y[trn_idx]
#     val_y = val_y[val_idx]    
    
    if predict:
        return train_X, val_X, test_X, train_y, val_y, tokenizer.word_index, tokenizer
    else:
        return train_X, val_X, train_y, val_y, tokenizer.word_index


def model_cnn(embedding_matrix):
    filter_sizes = [1, 2, 3, 5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                      kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def load_glove(word_index, embedding_fname='glove.840B.300d.txt'):
    EMBEDDING_FILE = './glove.840B.300d/' + embedding_fname

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8'))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in tqdm(word_index.items()):
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def predict_label(query='What is happiness?', maxlen = 70):
    from keras.models import load_model
    import pickle
    import time
    start = time.time()
    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    test_X = tokenizer.texts_to_sequences(query)
    test_X = pad_sequences(test_X, maxlen=maxlen)
    model = load_model('my_model2.h5')
    score = model.predict(test_X)
    print('took {:.2f} seconds to finish'.format(time.time() - start))
    if score[0] > best_thresh:
        return 1
    else:
        return 0


if __name__ == '__main__':

    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")

    positive = train_df[train_df.target == 1]
    del train_df, test_df


    # ### Baseline CNN
    max_features=95000
    maxlen=70
    embed_size=300

    train_X, val_X, test_X, train_y, val_y, word_index, tokenizer = data_preparation(predict=True)

    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    embedding_matrix1 = load_glove(word_index)
    # , embedding_fname='glove.6B.50d.txt'

    model = model_cnn(embedding_matrix1)

    for e in range(2):
        model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y))
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)

        best_thresh = 0.5
        best_score = 0.0
        for thresh in np.arange(0.1, 0.501, 0.01):
            thresh = np.round(thresh, 2)
            score = f1_score(val_y, (pred_val_y > thresh).astype(int))
            if score > best_score:
                best_thresh = thresh
                best_score = score

        print("Val F1 Score: {:.4f}".format(best_score))


    model.predict(test_X)

    model.save('my_model2.h5', 'w')

    model2 = load_model('my_model2.h5')

    model2.predict(test_X)

    predict_label()