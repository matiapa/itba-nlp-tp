import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def w2v_average_transform(X, sg, vec_size, window):       
    sentences = [txt.split(' ') for txt in X]

    embedding = Word2Vec(sentences=sentences, sg=sg, vector_size=vec_size, window=window, workers=4)

    new_X = []
    present, missing = 0, 0

    for txt in X:
        avg_coord = np.zeros((vec_size))
        words = txt.split(' ')
        
        for word in words:
            if word in embedding.wv:
                avg_coord += embedding.wv[word]
                present += 1
            else:
                missing += 1
        avg_coord /= len(words)
        
        new_X.append(avg_coord)
    
    return new_X

# ----------------------------------------------------------

corpus = pd.read_csv('../in/reduced_corpus.csv')
X = corpus['statement']
Y = corpus['verdict']

X = w2v_average_transform(X, sg=1, vec_size=100, window=5)

X = StandardScaler().fit_transform(X)

# ----------------------------------------------------------

print('Training...')

param_grid = {
    'C': np.logspace(-2, 1, 3), 'gamma': np.logspace(-6, 1, 3),
    'kernel': ['rbf'], 'cache_size': [1000]
}

cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=4, verbose=3)
grid.fit(X, Y)

print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)

# [CV 1/1] END C=0.01, cache_size=1000, gamma=0.0031622776601683794, kernel=rbf;, score=0.689 total time=  34.0s
# [CV 1/1] END C=0.01, cache_size=1000, gamma=1e-06, kernel=rbf;, score=0.590 total time=  36.6s
# [CV 1/1] END C=0.31622776601683794, cache_size=1000, gamma=1e-06, kernel=rbf;, score=0.590 total time=  37.0s
# [CV 1/1] END C=0.01, cache_size=1000, gamma=10.0, kernel=rbf;, score=0.590 total time=  50.7s
# [CV 1/1] END C=0.31622776601683794, cache_size=1000, gamma=0.0031622776601683794, kernel=rbf;, score=0.699 total time=  28.9s
# [CV 1/1] END C=10.0, cache_size=1000, gamma=1e-06, kernel=rbf;, score=0.678 total time=  37.5s
# [CV 1/1] END C=10.0, cache_size=1000, gamma=0.0031622776601683794, kernel=rbf;, score=0.714 total time=  34.8s
# [CV 1/1] END C=0.31622776601683794, cache_size=1000, gamma=10.0, kernel=rbf;, score=0.590 total time=  52.5s
# [CV 1/1] END C=10.0, cache_size=1000, gamma=10.0, kernel=rbf;, score=0.590 total time=  45.4s
# The best parameters are {'C': 10.0, 'cache_size': 1000, 'gamma': 0.0031622776601683794, 'kernel': 'rbf'} with a score of 0.71