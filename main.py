# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC

# Load the dataset
# data = pd.read_csv('fake_or_real_news.csv')

# # Convert labels to binary: 0 for REAL, 1 for FAKE
# data['fake'] = (data['label'] == 'FAKE').astype(int)
# data = data.drop('label', axis=1)

# # Split data into features (X) and labels (y)
# X, y = data['text'], data['fake']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize TF-IDF vectorizer
# vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# # Fit and transform the training data
# X_train_vectorized = vectorizer.fit_transform(X_train)

# # Transform the testing data
# X_test_vectorized = vectorizer.transform(X_test)

# # Initialize LinearSVC classifier
# clf = LinearSVC()

# # Train the classifier
# clf.fit(X_train_vectorized, y_train)

# # Evaluate the classifier
# accuracy = clf.score(X_test_vectorized, y_test)
# print("Accuracy:", accuracy)

# # Example of using the trained model to predict a new text
# new_text = "This is a sample news article."
# new_text_vectorized = vectorizer.transform([new_text])
# prediction = clf.predict(new_text_vectorized)
# print("Prediction:", prediction)



import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

data = pd.read_csv('fake_or_real_news.csv')

data['fake'] = data['label'].apply(lambda x: 0 if x == 'REAL' else 1)
data = data.drop('label', axis=1)
X, y = data['text'], data['fake']

X_train , X_test , y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)

accuracy = clf.score(X_test_vectorized, y_test)
print("Accuracy:", accuracy)

# Example of using the trained model to predict a new text
with open('mytext.txt', 'w', encoding='utf-8') as f:
    f.write(X_test.iloc[10])

with open('mytext.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vectorizer_text = vectorizer.transform([text])
prediction = clf.predict(vectorizer_text)

y_test.iloc[10]
