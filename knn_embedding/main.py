import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

clf = KNeighborsClassifier(n_neighbors=100, weights='uniform', metric='euclidean')
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

# ----------------------------------------------------------

cm = confusion_matrix(Y_test, Y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()