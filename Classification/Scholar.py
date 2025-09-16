#!/usr/bin/env python
# coding: utf-8
"""
Refactored script for TF-IDF vectorization and KMeans clustering of text from an
Excel file column. Converted from a Jupyter-exported script into a reusable module
with a small CLI.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------------
# Configuration and data loading (refactored)
# -------------------------------

def load_texts_from_excel(path: str, column: str) -> pd.Series:
    df = pd.read_excel(path)
    df = df.dropna().drop_duplicates()
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in {path}. Available: {list(df.columns)}")
    return df[column]

    # 1. Lowercasing
    text = text.lower()
    
    # 2. Removing non-alphanumeric characters (keeping spaces and letters only)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenization (splitting text into words)
    words = nltk.word_tokenize(text)
    
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # 5. Lemmatization (to reduce words to their base form)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # 6. Join words back into a string
    return ' '.join(words)

# Apply preprocessing to the text data
preprocessed_texts = texts.apply(preprocess_text)


# -------------------------------
# Vectorization (refactored)
# -------------------------------

def vectorize(texts: Sequence[str], max_df: float = 0.98, min_df: int = 2,
              ngram_range: Tuple[int, int] = (1, 2)) -> Tuple[object, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range,
        stop_words='english'
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


# # Question 2

# In[18]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Elbow determination (refactored)
# -------------------------------

def elbow_k(X, k_min: int = 1, k_max: int = 20, plot: bool = True, random_state: int = 42) -> int:
    k_range = range(k_min, k_max)
    inertia = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    if len(inertia) < 2:
        elbow_point = k_min
    else:
        inertia_diff = np.diff(inertia)
        elbow_point = int(np.argmax(inertia_diff) + k_min)
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(list(k_range), inertia, marker='o', linestyle='--')
        plt.axvline(x=elbow_point, color='r', linestyle='--', label=f"Elbow Point: K={elbow_point}")
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia')
        plt.xticks(list(k_range))
        plt.grid(True)
        plt.legend()
        plt.show()
    print(f"Optimal number of clusters (elbow point): {elbow_point}")
    return elbow_point


# In[33]:


# -------------------------------
# Clustering and reporting (refactored)
# -------------------------------

def fit_kmeans(X, n_clusters: int, random_state: int = 42) -> KMeans:
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(X)


def safe_silhouette(X, labels: Sequence[int]) -> Optional[float]:
    try:
        if len(set(labels)) < 2:
            return None
        return float(silhouette_score(X, labels))
    except Exception:
        return None


# In[34]:


# -------------------------------
# Top terms per cluster (refactored)
# -------------------------------

def top_terms_per_cluster(kmeans: KMeans, vectorizer: TfidfVectorizer, top_n: int = 20) -> List[List[str]]:
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    tops: List[List[str]] = []
    for i in range(kmeans.n_clusters):
        tops.append([terms[ind] for ind in order_centroids[i, :top_n]])
    return tops


# -------------------------------
# PCA and CLI (refactored)
# -------------------------------

# In[36]:


from sklearn.decomposition import PCA


# In[37]:


pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X.toarray())


# In[38]:

# End of deprecated notebook section. Refactored CLI below.

def plot_pca(X, labels: Sequence[int], title: str = "PCA of Clusters in 2D Space") -> None:
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        pca = PCA(n_components=2)
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        X_pca = pca.fit_transform(X_dense)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o')
        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label="Cluster")
        plt.show()
    except Exception:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TF-IDF + KMeans clustering for Excel text column")
    parser.add_argument("--input", required=True, help="Path to Excel file")
    parser.add_argument("--column", required=True, help="Column name containing text (e.g., ACTIVITIES)")
    parser.add_argument("--clusters", type=int, default=8, help="Number of clusters. Use 0 for elbow-based auto")
    parser.add_argument("--plot", action="store_true", help="Show elbow and PCA plots")
    parser.add_argument("--max-df", type=float, default=0.98)
    parser.add_argument("--min-df", type=int, default=2)
    args = parser.parse_args(argv)

    # Ensure NLTK resources
    for res_name, res_path in [("punkt", "tokenizers/punkt"), ("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(res_path)
        except LookupError:
            try:
                nltk.download(res_name, quiet=True)
            except Exception:
                pass

    texts = load_texts_from_excel(args.input, args.column).astype(str)
    stop_words = frozenset(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    token_pattern = re.compile(r"[^a-zA-Z\s]")

    def preprocess_text(text: str) -> str:
        t = token_pattern.sub(" ", text.lower())
        words = nltk.word_tokenize(t)
        words = [w for w in words if w and w not in stop_words]
        words = [lemmatizer.lemmatize(w) for w in words]
        return " ".join(words)

    preprocessed = texts.apply(preprocess_text)

    X, vectorizer = vectorize(preprocessed, max_df=args.max_df, min_df=args.min_df)

    print("Features (Top 20):", vectorizer.get_feature_names_out()[:20])
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    print(tfidf_df.head())
    print(tfidf_df.shape)

    if args.clusters and args.clusters > 0:
        n_clusters = args.clusters
        if args.plot:
            elbow_k(X, plot=True)
    else:
        n_clusters = max(2, elbow_k(X, plot=args.plot))

    kmeans = fit_kmeans(X, n_clusters)
    labels = list(kmeans.labels_)

    sil = safe_silhouette(X, labels)
    if sil is not None:
        print(f"Silhouette Score: {sil}")
    else:
        print("Silhouette Score: N/A (requires at least 2 clusters and >1 label)")

    tops = top_terms_per_cluster(kmeans, vectorizer, top_n=20)
    for i, words in enumerate(tops, start=1):
        print(f"Cluster {i} top 10 words:")
        print(", ".join(words[:20]))
        print("\n")

    if args.plot:
        plot_pca(X, labels)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
[EOF](No newline at end of file)
X_pca.shape


# In[40]:


# Step 4: Plot the data points in 2D PCA space with clusters as colors
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
plt.title("PCA of Clusters in 2D Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.show()


# In[ ]:





# coding: utf-8

# ## Import Libraries

# In[2]:


import os
from pathlib import Path
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def _resolve_score_xlsx():
    env = os.environ.get('SCORE_XLSX')
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path('Score.xlsx'),
        Path('data/Score.xlsx'),
        script_dir / 'Score.xlsx',
        script_dir / 'data' / 'Score.xlsx',
        Path.cwd() / 'Score.xlsx',
        Path.cwd() / 'data' / 'Score.xlsx',
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Score.xlsx not found. Set SCORE_XLSX environment variable to the file path "
        "or place Score.xlsx in the project root or a data/ directory."
    )


# ## Loading data

# In[3]:


# Resolve data file path robustly (supports SCORE_XLSX env var and common locations)
_score_path = _resolve_score_xlsx()
df = pd.read_excel(_score_path)
df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df['ACTIVITIES'].duplicated().sum()


# In[6]:


df = df.dropna().drop_duplicates()


# In[7]:


df


# In[8]:


texts = df['ACTIVITIES']


# In[9]:


texts


# In[10]:


def preprocess_text(text):
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Removing non-alphanumeric characters (keeping spaces and letters only)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenization (splitting text into words)
    words = nltk.word_tokenize(text)
    
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # 5. Lemmatization (to reduce words to their base form)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # 6. Join words back into a string
    return ' '.join(words)

# Apply preprocessing to the text data
preprocessed_texts = texts.apply(preprocess_text)


# In[11]:


preprocessed_texts


# In[12]:


vectorizer = TfidfVectorizer(
    max_df= 0.98,  # Ignore words that appear in more than 85% of documents
    min_df=2,     # Ignore words that appear in less than 5 documents
    ngram_range=(1, 2),  # Use unigrams
    stop_words='english'  # Remove common English stopwords
)


# In[13]:


X = vectorizer.fit_transform(preprocessed_texts)


# In[14]:


X


# In[15]:


print("Features (Top 20):", vectorizer.get_feature_names_out()[:20])


# In[16]:


# Convert the matrix to a DataFrame
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Display the top few rows of the TF-IDF DataFrame
print(tfidf_df.head())


# In[17]:


tfidf_df.shape


# # Question 2

# In[18]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt


# In[19]:


# Range of clusters to try
k_range = range(1, 20)

# List to store inertia values for each K
inertia = []

# Calculate inertia for each K
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Set n_jobs=1 to disable parallelism
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Calculate the first derivative (difference between successive inertia values)
inertia_diff = np.diff(inertia)

# Find the index where the first derivative decreases the most (elbow point)
elbow_point = np.argmax(inertia_diff) + 1  # Adding 1 because the first diff is one less than the actual K

# Plot the Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.axvline(x=elbow_point, color='r', linestyle='--', label=f"Elbow Point: K={elbow_point}")
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.legend()
plt.show()

print(f"Optimal number of clusters (elbow point): {elbow_point}")


# In[33]:


# Clustering using K-Means
num_clusters = 8  # Set the number of clusters (You can adjust this based on evaluation metrics)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Evaluate clustering using silhouette score
sil_score = silhouette_score(X, labels)

# Show the results
print(f"Silhouette Score: {sil_score}")


# In[34]:


# Extract the top 10 words for each cluster
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

# Display the top 10 words in each cluster
for i in range(num_clusters):
    print(f"Cluster {i+1} top 10 words:")
    top_words = [terms[ind] for ind in order_centroids[i, :20]]
    print(", ".join(top_words))
    print("\n")


# In[35]:

df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df['ACTIVITIES'].duplicated().sum()


# In[6]:


df = df.dropna().drop_duplicates()


# In[7]:


df


# In[8]:


texts = df['ACTIVITIES']


# In[9]:


texts


# In[10]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Pre-computed resources (built once at import time)
# ---------------------------------------------------------------------------

_NON_ALPHA_RE = re.compile(r'[^a-zA-Z\s]')                 # compiled pattern
_STOP_WORDS   = set(stopwords.words('english'))            # stop-word set
_LEMMATIZER   = WordNetLemmatizer()                        # shared lemmatizer
_TOKENIZE     = nltk.word_tokenize                         # alias for speed

# ---------------------------------------------------------------------------
# Text-preprocessing helper
# ---------------------------------------------------------------------------

def preprocess_text(
    text: str,
    *,
    _pattern: re.Pattern = _NON_ALPHA_RE,
    _tokenizer           = _TOKENIZE,
    _stop_words: set     = _STOP_WORDS,
    _lemmatizer          = _LEMMATIZER
) -> str:
    """
    Normalise, clean, tokenise, remove stop-words, lemmatise and re-join text.

    Steps:
        1. Lower-case
        2. Strip non-alphabetic characters
        3. Tokenise
        4. Drop English stop-words
        5. Lemmatise
    """
    # 1–2. lower-case & regex cleaning
    text = _pattern.sub('', text.lower())

    # 3–5. tokenise → filter stop-words → lemmatise (single pass)
    words = (
        _lemmatizer.lemmatize(token)
        for token in _tokenizer(text)
        if token not in _stop_words
    )

    # Re-join tokens
    return ' '.join(words)
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize NLTK resources safely (download on first use if missing)
try:
    _STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    _STOP_WORDS = set(stopwords.words('english'))

# Ensure WordNet data is available for lemmatization
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    # Optional multilingual WordNet data
    try:
        nltk.download('omw-1.4', quiet=True)
    except Exception:
        pass

_LEMMATIZER = WordNetLemmatizer()


# ## Loading data

# In[3]:


df = pd.read_excel('C:\\Users\\Maclan\\OneDrive\\Desktop\\Score.xlsx')
df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df['ACTIVITIES'].duplicated().sum()


# In[6]:


df = df.dropna().drop_duplicates()


# In[7]:


df


# In[8]:


texts = df['ACTIVITIES']


# In[9]:


texts


# In[10]:


def preprocess_text(text):
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Removing non-alphanumeric characters (keeping spaces and letters only)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenization (splitting text into words)
    # Since we've removed non-letters and kept spaces, a simple split is sufficient and
    # avoids requiring the punkt tokenizer resource.
    words = text.split()
    
    # 4. Remove stopwords (use preloaded set for efficiency)
    words = [word for word in words if word not in _STOP_WORDS]
    
    # 5. Lemmatization (to reduce words to their base form) using preloaded lemmatizer
    words = [_LEMMATIZER.lemmatize(word) for word in words]
    
    # 6. Join words back into a string
    return ' '.join(words)

# Apply preprocessing to the text data
preprocessed_texts = texts.apply(preprocess_text)


# In[11]:


preprocessed_texts


# In[12]:


vectorizer = TfidfVectorizer(
    max_df= 0.98,  # Ignore words that appear in more than 85% of documents
    min_df=2,     # Ignore words that appear in less than 5 documents
    ngram_range=(1, 2),  # Use unigrams
    stop_words='english'  # Remove common English stopwords
)


# In[13]:


X = vectorizer.fit_transform(preprocessed_texts)


# In[14]:


X


# In[15]:


print("Features (Top 20):", vectorizer.get_feature_names_out()[:20])


# In[16]:


# Convert the matrix to a DataFrame
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Display the top few rows of the TF-IDF DataFrame
print(tfidf_df.head())


# In[17]:


tfidf_df.shape


# # Question 2

# In[18]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt


# In[19]:


# Range of clusters to try
k_range = range(1, 20)

# List to store inertia values for each K
inertia = []

# Calculate inertia for each K
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Set n_jobs=1 to disable parallelism
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Calculate the first derivative (difference between successive inertia values)
inertia_diff = np.diff(inertia)

# Find the index where the first derivative decreases the most (elbow point)
elbow_point = np.argmax(inertia_diff) + 1  # Adding 1 because the first diff is one less than the actual K

# Plot the Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.axvline(x=elbow_point, color='r', linestyle='--', label=f"Elbow Point: K={elbow_point}")
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.legend()
plt.show()

print(f"Optimal number of clusters (elbow point): {elbow_point}")


# In[33]:


# Clustering using K-Means
num_clusters = 8  # Set the number of clusters (You can adjust this based on evaluation metrics)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Evaluate clustering using silhouette score
sil_score = silhouette_score(X, labels)

# Show the results
print(f"Silhouette Score: {sil_score}")


# In[34]:


# Extract the top 10 words for each cluster
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

# Display the top 10 words in each cluster
for i in range(num_clusters):
    print(f"Cluster {i+1} top 10 words:")
    top_words = [terms[ind] for ind in order_centroids[i, :20]]
    print(", ".join(top_words))
    print("\n")


# In[35]:


labels.shape


# ## Question 03

    
    # 6. Join words back into a string
    return ' '.join(words)

# Apply preprocessing to the text data
preprocessed_texts = texts.apply(preprocess_text)


# In[11]:


preprocessed_texts


# In[12]:


vectorizer = TfidfVectorizer(
    max_df=0.98,  # Ignore words that appear in more than 98% of documents
    min_df=2,  # Ignore words that appear in less than 2 documents
    ngram_range=(1, 2),  # Use unigrams and bigrams
    stop_words='english'  # Remove common English stopwords
)


# In[13]:


X = vectorizer.fit_transform(preprocessed_texts)


# In[14]:


X


# In[15]:


print("Features (Top 20):", vectorizer.get_feature_names_out()[:20])


# In[16]:


# Convert the matrix to a DataFrame
#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[2]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# ## Loading data

# In[3]:


df = pd.read_excel('C:\\Users\\Maclan\\OneDrive\\Desktop\\Score.xlsx')
df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df['ACTIVITIES'].duplicated().sum()


# In[6]:


df = df.dropna().drop_duplicates()


# In[7]:


df


# In[8]:


texts = df['ACTIVITIES']


# In[9]:


texts


# In[10]:


def preprocess_text(text):
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Removing non-alphanumeric characters (keeping spaces and letters only)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenization (splitting text into words)
    words = nltk.word_tokenize(text)
    
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # 5. Lemmatization (to reduce words to their base form)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # 6. Join words back into a string
    return ' '.join(words)

# Apply preprocessing to the text data
preprocessed_texts = texts.apply(preprocess_text)


# In[11]:


preprocessed_texts


# In[12]:


vectorizer = TfidfVectorizer(
    max_df= 0.98,  # Ignore words that appear in more than 85% of documents
    min_df=2,     # Ignore words that appear in less than 5 documents
    ngram_range=(1, 2),  # Use unigrams
    stop_words='english'  # Remove common English stopwords
)


# In[13]:


X = vectorizer.fit_transform(preprocessed_texts)


# In[14]:


X


# In[15]:


print("Features (Top 20):", vectorizer.get_feature_names_out()[:20])


# In[16]:


# Convert the matrix to a sparse DataFrame to avoid dense memory usage
tfidf_df = pd.DataFrame.sparse.from_spmatrix(X, columns=vectorizer.get_feature_names_out())

# Display the top few rows in a dense format for readability (only the head is densified)
print(tfidf_df.head().sparse.to_dense())


# In[17]:


tfidf_df.shape


# # Question 2

# In[18]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt


# In[19]:


# Range of clusters to try
k_range = range(1, 20)

# List to store inertia values for each K
inertia = []

# Calculate inertia for each K
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Set n_jobs=1 to disable parallelism
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Calculate the first derivative (difference between successive inertia values)
inertia_diff = np.diff(inertia)

# Find the index where the first derivative decreases the most (elbow point)
elbow_point = np.argmax(inertia_diff) + 1  # Adding 1 because the first diff is one less than the actual K

# Plot the Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.axvline(x=elbow_point, color='r', linestyle='--', label=f"Elbow Point: K={elbow_point}")
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.legend()
plt.show()

print(f"Optimal number of clusters (elbow point): {elbow_point}")


# In[33]:


# Clustering using K-Means
num_clusters = 8  # Set the number of clusters (You can adjust this based on evaluation metrics)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Evaluate clustering using silhouette score
sil_score = silhouette_score(X, labels)

# Show the results
print(f"Silhouette Score: {sil_score}")


# In[34]:


# Extract the top 10 words for each cluster
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

# Display the top 10 words in each cluster
for i in range(num_clusters):
    print(f"Cluster {i+1} top 10 words:")
    top_words = [terms[ind] for ind in order_centroids[i, :20]]
    print(", ".join(top_words))
    print("\n")


# In[35]:


labels.shape


# ## Question 03

# In[36]:


from sklearn.decomposition import PCA


# In[37]:


pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X.toarray())


# In[38]:


X_pca


# In[39]:


# In[17]:


tfidf_df.shape


# # Question 2

# In[18]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt


# In[19]:


# Range of clusters to try
k_range = range(1, 20)

# List to store inertia values for each K
inertia = []

# Calculate inertia for each K
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Set n_jobs=1 to disable parallelism
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Calculate the first derivative (difference between successive inertia values)
inertia_diff = np.diff(inertia)

# Find the index where the first derivative decreases the most (elbow point)
elbow_point = np.argmax(inertia_diff) + 1  # Adding 1 because the first diff is one less than the actual K

# Plot the Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.axvline(x=elbow_point, color='r', linestyle='--', label=f"Elbow Point: K={elbow_point}")
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.legend()
plt.show()

print(f"Optimal number of clusters (elbow point): {elbow_point}")


# In[33]:


# Clustering using K-Means
num_clusters = 8  # Set the number of clusters (You can adjust this based on evaluation metrics)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Evaluate clustering using silhouette score
sil_score = silhouette_score(X, labels)

# Show the results
print(f"Silhouette Score: {sil_score}")


# In[34]:


# Extract the top 10 words for each cluster
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

# Display the top 10 words in each cluster
for i in range(num_clusters):
    print(f"Cluster {i+1} top 10 words:")
    top_words = [terms[ind] for ind in order_centroids[i, :20]]
    print(", ".join(top_words))
    print("\n")


# In[35]:


labels.shape


# ## Question 03

# In[36]:


from sklearn.decomposition import PCA


# In[37]:


# ---------------------------------------------------------------------------
# Dimensionality Reduction
# ---------------------------------------------------------------------------
# NOTE:
# 1. Converting a large sparse TF-IDF matrix to a dense NumPy array with
#    `.toarray()` can easily exhaust system memory.
# 2. Scikit-learn’s PCA implementation only accepts dense input, whereas
#    TruncatedSVD can work directly on sparse matrices while producing an
#    output that is *identical* to PCA when `n_components < min(n_samples,
#    n_features)`.
# 3. To keep the original behaviour for small datasets (where the dense
#    conversion is harmless) yet remain memory-safe for larger corpora, we
#    automatically choose between `PCA` and `TruncatedSVD` based on whether
#    the input is sparse.
#
# The helper below encapsulates that logic and returns both the transformed
# matrix and the fitted dimensionality-reduction model.

from typing import Tuple
from scipy.sparse import issparse
from sklearn.decomposition import PCA, TruncatedSVD


def reduce_dimensionality(
        X,
        n_components: int = 2,
        *,
        random_state: int = 42,
        force_dense: bool = False,
) -> Tuple["np.ndarray", "BaseEstimator"]:
    """
    Reduce dimensionality of `X` to `n_components`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to be reduced.
    n_components : int, default=2
        Target dimensionality.
    random_state : int, default=42
        Random seed for reproducibility.
    force_dense : bool, default=False
        If True, always use PCA on the dense representation of `X`.
        If False, fall back to TruncatedSVD when `X` is sparse.

    Returns
    -------
    X_reduced : ndarray of shape (n_samples, n_components)
        The dimensionality-reduced data.
    reducer : fitted PCA or TruncatedSVD instance
    """
    is_sparse = issparse(X)
    if force_dense or not is_sparse:
        # Use standard PCA on a dense array
        reducer = PCA(n_components=n_components, random_state=random_state)
        X_input = X.toarray() if hasattr(X, "toarray") else X
    else:
        # Memory-efficient alternative for sparse input
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
        X_input = X  # SVD accepts sparse matrices directly

    X_reduced = reducer.fit_transform(X_input)
    return X_reduced, reducer


# ---------------------------------------------------------------------------
# Apply the helper to the TF-IDF feature matrix
# ---------------------------------------------------------------------------
#  - `X` is produced earlier via `TfidfVectorizer`.
#  - We retain the variable name `X_pca` for compatibility with downstream
#    code (plots, analyses, etc.), regardless of the reducer actually used.

X_pca, pca = reduce_dimensionality(X, n_components=2)

# `X_pca` now contains the 2-D embedding that can be used for visualisation,
# clustering diagnostics, or any other downstream task.

# In[38]:


X_pca


# In[39]:


X_pca.shape


# In[40]:


# Step 4: Plot the data points in 2D PCA space with clusters as colors
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
plt.title("PCA of Clusters in 2D Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.show()


# In[ ]: