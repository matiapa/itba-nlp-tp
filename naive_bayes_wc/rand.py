import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

corpus = pd.read_csv('../in/reduced_corpus.csv')
X = corpus['statement']
Y = corpus['verdict']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# ----------------------------------------------------------

pipeline = Pipeline([
    ("vect", CountVectorizer()),
    ("clf", ComplementNB()),
])

parameter_grid = {
    "vect__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),
    "vect__min_df": (1, 3, 5, 10),
    "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
    "clf__alpha": np.logspace(-6, 6, 13),
}

random_search = RandomizedSearchCV(
    estimator=pipeline, param_distributions=parameter_grid,
    n_iter=50, n_jobs=4, random_state=0, verbose=1
)

random_search.fit(X_train, Y_train)

# ----------------------------------------------------------

print("Best parameters combination found:")
best_parameters = random_search.best_estimator_.get_params()
for param_name in sorted(parameter_grid.keys()):
    print(f"{param_name}: {best_parameters[param_name]}")

Y_pred = random_search.predict(X_test)

# ----------------------------------------------------------

cm = confusion_matrix(Y_test, Y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()