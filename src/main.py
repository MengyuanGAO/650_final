from keras.models import load_model
import pickle
import time
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize
import numpy as np
import nltk
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))

def modelInit():
    model = load_model('../data/CNN_flip.h5')
    return model

def queryPredict(query, model, maxlen=70, best_thresh=0.5):
    # model = load_model('../data/my_model2.h5')
    with open('../data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    test_X = tokenizer.texts_to_sequences([query])
    test_X = pad_sequences(test_X, maxlen=maxlen)
    score = model.predict(test_X)
    if score[0] > best_thresh:
        return 1
    else:
        return 0


def scoring_words(query='Has the United States become the largest dictatorship in the world?'):
    thres = 1.0
    # preprocessing
    tokens = word_tokenize(query.lower())

    with open('../data/lg_score.pickle', 'rb') as handle:
        b = pickle.load(handle)

    scores = [b[t] if t in b else 0 for t in tokens]

    #     import numpy as np
    #     arr = np.array(scores)
    #     indices = arr.argsort()[-5:][::-1]
    indices = [scores.index(ii) for ii in scores if (ii >= thres) and (tokens[scores.index(ii)] not in stopwords)]
    words = list(set([tokens[i] for i in indices]))
    print(words)
    ans = []
    for ind, ii in enumerate(query.lower().split()):
        for jj in words:
            if jj in ii and len(ii) - len(jj) <= 1:
                ans.append(ind)
    return ans

if __name__ == '__main__':
    print(queryPredict('Has the United States become the largest dictatorship in the world?', modelInit()))
    print(scoring_words("Why don't USA citizens realize that Trump is rapidly doing what terrorists could not, i.e., push the country towards irrevocable catastrophe?"))