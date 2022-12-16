import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


corpus = pd.read_csv('../in/reduced_corpus.csv')
X = corpus['statement']
Y = corpus['verdict']

vectorizer = CountVectorizer(ngram_range=(1,4))
X = vectorizer.fit_transform(corpus['statement'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# ----------------------------------------------------------

clf = RandomForestClassifier(max_depth=50, random_state=0)
clf.fit(X_train, Y_train)

# ----------------------------------------------------------

Y_pred = clf.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()