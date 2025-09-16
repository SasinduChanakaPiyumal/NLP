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
# Classification/Scholar.py
# -------------------------
# NOTE: Only the relevant section (lines 73-81) has been refactored.
#       Surrounding code is kept intact to avoid breaking external consumers.

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# One-time initialisation of heavy NLTK resources
# ---------------------------------------------------------------------------
# These downloads are guarded so they only run when the resources are missing.
for resource in ("punkt", "stopwords", "wordnet"):
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

# Caching expensive objects improves performance when the function is called
# repeatedly (e.g., inside a pandas .apply()).
_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Text-preprocessing utility
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """
    Normalise and clean a raw text string.

    Steps
    -----
    1. Lower-case conversion.
    2. Strip non-alphabetic characters (numbers, punctuation, etc.).
    3. Tokenisation.
    4. Stop-word removal.
    5. Lemmatisation.
    6. Re-join tokens into a single string.

    Parameters
    ----------
    text : str
        The raw input text.

    Returns
    -------
    str
        A processed, tokenised string ready for feature extraction.
    """
    # 1. Lowercasing
    text = text.lower()

    # 2. Removing non-alphabetic characters while keeping spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 3. Tokenisation
    words = nltk.word_tokenize(text)

    # -----------------------------------------------------------------------
    #  Refactored block (originally lines 73-81)
    # -----------------------------------------------------------------------
    # 4. Stop-word removal (using cached global set)
    words = [word for word in words if word not in _STOP_WORDS]

    # 5. Lemmatisation (using cached global lemmatiser)
    words = [_LEMMATIZER.lemmatize(word) for word in words]
    # -----------------------------------------------------------------------

    # 6. Join tokens back into a single space-separated string
    return " ".join(words)
    
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

# In[36]:


from sklearn.decomposition import PCA


# In[37]:


pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X.toarray())


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



