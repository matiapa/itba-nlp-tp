import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


df = pd.read_json('in/politifact_factcheck_data.json', lines=True)

df = df.drop(columns=['statement_originator', 'statement_date', 'statement_source', 'factchecker', 'factcheck_date', 'factcheck_analysis_link'])

sia = SentimentIntensityAnalyzer()

df['sentiment'] = df.apply(lambda row: sia.polarity_scores(row['statement'])['compound'], axis=1)

df.to_csv('in/data.csv', index=False)