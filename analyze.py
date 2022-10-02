import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gensim

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.data import find
from sklearn.decomposition import PCA


df = pd.read_csv('in/data.csv')


def sentiment_vs_verdict_heatmaps():

    heatmap = np.zeros((5,5))

    verdict_map = {'true': 0, 'mostly-true': 1, 'half-true': 2, 'mostly-false': 3, 'false': 4, 'pants-fire': 4}

    for _, row in df.iterrows():

        s = 0 if row['sentiment'] > 0.5 else (1 if row['sentiment'] > 0 else (2 if row['sentiment'] == 0 else(3 if row['sentiment'] > -0.5 else 4)))

        v = verdict_map[row['verdict']]

        heatmap[s][v] += 1

    for s in range(5):
        count = sum(heatmap[s])
        for v in range(5):
            heatmap[s][v] /= count

    # for v in range(5):
    #     count = sum(heatmap[:,v])
    #     for s in range(5):
    #         heatmap[s][v] /= count

    sns.heatmap(heatmap, xticklabels=['true', 'mostly-true', 'half-true', 'mostly-false', 'false'], yticklabels=['positive', 'mostly-positive', 'neutral', 'mostly-negative', 'negative'])
    plt.show()


def words_sentiment_trustness():

    words = {}

    trustness_map = {'true': 1, 'mostly-true': 0.5, 'half-true': 0, 'mostly-false': -0.5, 'false': -1, 'pants-fire': -1}

    lemmatizer = WordNetLemmatizer()

    for _, row in df.iterrows():

        for word in word_tokenize(row['statement']):

            word = lemmatizer.lemmatize(word)

            if word not in words:
                words[word] = {'count': 0, 'sentiment': 0, 'trustness': 0}

            words[word]['count'] += 1
            words[word]['sentiment'] += row['sentiment']
            words[word]['trustness'] += trustness_map[row['verdict']]

    filtered_words = {}
    for word in words:
        if words[word]['count'] > 10:
            words[word]['sentiment'] /= words[word]['count']
            words[word]['trustness'] /= words[word]['count']
            filtered_words[word] = words[word]

    return filtered_words


def sentiment_trustness_word_rank():
    words = words_sentiment_trustness()

    sentiment_rank = sorted(words.items(), key = lambda i : i[1]['sentiment'], reverse=True)
    trustness_rank = sorted(words.items(), key = lambda i : i[1]['trustness'], reverse=True)

    print(sentiment_rank)
    print(trustness_rank)


def pruned_word2vec():

    word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

    # Load words coordinates into a grid

    w2v_words, count = [], 0
    w2v_coords = np.zeros(shape=(len(model), len(model['university'])))

    for term in model.index_to_key:
        print(f'{count}/{len(model)}')
        w2v_coords[count] = model[term]
        w2v_words.append(term)
        count+= 1

    # Reduce dimensionality with PCA

    print('PCA')
    pca = PCA(n_components=2)
    w2v_coords = pca.fit_transform(w2v_coords)

    # Return as a dictionary

    w2v = {}
    for i in range(len(w2v_words)):
        w2v[w2v_words[i]] = w2v_coords[i]

    return w2v


def sentiment_trustness_word2vec_scatterplot():

    w2v = pruned_word2vec()
    words = words_sentiment_trustness()
    X, Y, S, T = [], [], [], []

    for word in words:
        if word in w2v:
            X.append(w2v[word][0])
            Y.append(w2v[word][1])
            S.append(words[word]['sentiment'])
            T.append(words[word]['trustness'])

    sns.scatterplot(X, Y, hue = S)
    # sns.scatterplot(X, Y, hue = T)
    
    plt.show()