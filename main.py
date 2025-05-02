import pandas as pd
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("data/fake_or_real_news.csv")

# Drop unwanted columns
df.drop(columns=["Unnamed: 0"], inplace=True)

# Combine title and text
df["content"] = df["title"] + " " + df["text"]

# Encode labels
df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

# Preprocessing: remove punctuation
def clean_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower()

df["content"] = df["content"].apply(clean_text)

# Features and labels
X = df["content"]
y = df["label"]

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and vectorizer
with open("models/fake_news_model.pkl", "wb") as f:
    pickle.dump((model, tfidf), f)

# Prediction function
def predict_news(text):
    with open("models/fake_news_model.pkl", "rb") as f:
        model, tfidf = pickle.load(f)
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "REAL" if prediction == 1 else "FAKE"

# Test
sample = "The government has passed a new bill to support farmers."
print("Prediction:", predict_news(sample))
