# import pandas as pd
# import re
# import string
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report

# # Text preprocessing function
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'\d+', '', text)
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # Load dataset
# df = pd.read_csv('data/combined_emails_with_natural_pii.csv')  # adjust path if needed
# print(df.columns)
# print(df.head())

# df.dropna(subset=['email', 'type'], inplace=True)

# # Clean text data
# df['cleaned_text'] = df['email'].apply(clean_text)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     df['cleaned_text'], df['type'], test_size=0.2, random_state=42
# )

# # Create a pipeline with TF-IDF + Naive Bayes
# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
#     ('clf', MultinomialNB())
# ])

# # Train the model
# pipeline.fit(X_train, y_train)

# # Evaluate the model
# y_pred = pipeline.predict(X_test)
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # Save the model
# with open('email_classifier.pkl', 'wb') as model_file:
#     pickle.dump(pipeline, model_file)

import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

# Load dataset
df = pd.read_csv('data/combined_emails_with_natural_pii.csv')  # adjust path if needed
# print(df.columns)
# print(df.head())

df.dropna(subset=['email', 'type'], inplace=True)

# Clean text data
df['cleaned_text'] = df['email'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['type'], test_size=0.2, random_state=42
)

# Create a pipeline with TF-IDF + Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 2), max_features=5000)),
    ('clf', MultinomialNB())
])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__max_features': [5000, 10000, 20000],
    'clf__alpha': [0.1, 0.5, 1.0]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Evaluate the model
y_pred = grid_search.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
with open('email_classifier.pkl', 'wb') as model_file:
    pickle.dump(grid_search.best_estimator_, model_file)
