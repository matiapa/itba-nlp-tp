import string
import pandas as pd
import warnings
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

warnings.filterwarnings("ignore", category=FutureWarning) 

df = pd.read_json('in/politifact.json', lines=True)

full_corpus = pd.DataFrame(columns=['statement', 'verdict', 'sentiment'])
redu_corpus = pd.DataFrame(columns=['statement', 'verdict', 'sentiment'])

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

for _, article in tqdm(df.iterrows()):
    text = article['statement']

    text.lower()

    text = "".join([char for char in text if char not in string.punctuation])

    text = ' '.join([
        lemmatizer.lemmatize(word) for word in text.split(" ") if word not in stop_words
    ])

    verdict = article['verdict']

    sentiment = sia.polarity_scores(text)['compound']

    processed_article = {'statement': text, 'verdict': verdict, 'sentiment': sentiment}

    full_corpus = full_corpus.append(processed_article, ignore_index=True)

    if article['verdict'] not in ['half-true', 'mostly-false']:
        article['verdict'] = 'true' if verdict in ['true', 'mostly-true'] else 'false'
        redu_corpus = redu_corpus.append(processed_article, ignore_index=True)

full_corpus.to_csv('in/full_corpus.csv', index=False)
redu_corpus.to_csv('in/reduced_corpus.csv', index=False)