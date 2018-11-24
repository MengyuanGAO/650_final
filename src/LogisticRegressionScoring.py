import pickle
from nltk import word_tokenize
import numpy as np

def scoring_words(query='Has the United States become the largest dictatorship in the world?'):
    # preprocessing

    tokens = word_tokenize(query.lower())
#     print(tokens)
    
    with open('../data/lg_score.pickle', 'rb') as handle:
        b = pickle.load(handle)
        
    scores = [b[t] if t in b else 0 for t in tokens]


    arr = np.array(scores)
    indices = arr.argsort()[-5:][::-1]
    
#     print([tokens[i] for i in indices])
#     print(indices)
    
    return indices

if __name__ == '__main__':
    print(scoring_words("Why don't USA citizens realize that Trump is rapidly doing what terrorists could not, i.e., push the country towards irrevocable catastrophe?"))

