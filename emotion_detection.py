import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# --- 1. Download NLTK resources ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# --- 2. Initialize tools ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# --- 3. Text preprocessing ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('[^a-zA-Z\s]', '', text)
    words = text.split()
    cleaned_words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(cleaned_words)

# --- 4. Load dataset ---
try:
    df = pd.read_csv('emotion.csv')
    df = df.dropna()
    df['Cleaned_Text'] = df['Text'].apply(clean_text)
except FileNotFoundError:
    print("Error: emotion.csv not found!")
    exit()

# --- 5. TF-IDF ---
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X = tfidf.fit_transform(df['Cleaned_Text'])
y = df['Emotion']

# --- 6. Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)

# --- 7. Model Comparison ---
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()
}

results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    trained_models[name] = model
    
    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# --- 8. Plot comparison ---
plt.figure(figsize=(8,5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Comparison (Accuracy)")
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.show()

# --- 9. Choose best model ---
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print(f"\n🔥 Best Model: {best_model_name}")

# --- 10. Confusion Matrix ---
y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best, normalize='true')

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=best_model.classes_,
            yticklabels=best_model.classes_)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix ({best_model_name})")
plt.show()

# --- 11. Save model ---
joblib.dump(best_model, "best_emotion_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\nModel saved successfully!")

# --- 12. Prediction function ---
def predict_emotion(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    return best_model.predict(vectorized)[0]

# --- 13. Test prediction ---
test_sentence = "Oh great, another meeting that could have been an email."
print("\n--- SAMPLE PREDICTION ---")
print(f"Input: {test_sentence}")
print(f"Prediction: {predict_emotion(test_sentence)}")