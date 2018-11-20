from keras.models import load_model
import pickle
import time
from keras.preprocessing.sequence import pad_sequences

def modelInit():
    model = load_model('../data/my_model2.h5')
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

if __name__ == '__main__':
    print(queryPredict('Has the United States become the largest dictatorship in the world?', modelInit()))