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
import os
import sys


# ## Loading data

# In[3]:


# Configuration: Use environment variable or default path
excel_file_path = os.getenv('SCHOLAR_DATA_FILE', 'Score.xlsx')

try:
    df = pd.read_excel(excel_file_path)
    print(f"Successfully loaded data from {excel_file_path}")
    df.head()
except FileNotFoundError:
    print(f"Error: File '{excel_file_path}' not found.")
    print("Please ensure the file exists or set the SCHOLAR_DATA_FILE environment variable.")
    sys.exit(1)
except PermissionError:
    print(f"Error: Permission denied when accessing '{excel_file_path}'.")
    print("Please check file permissions.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading Excel file '{excel_file_path}': {str(e)}")
    print("Please check that the file is a valid Excel file and not corrupted.")
    sys.exit(1)


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

# Ensure required NLTK resources are available
def ensure_nltk_resources():
    """Download required NLTK resources if not available."""
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet')
    ]
    
    for name, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {name}")
            try:
                nltk.download(name, quiet=True)
            except Exception as e:
                print(f"Error downloading NLTK resource '{name}': {str(e)}")
                print("Please check your internet connection or download manually.")
                sys.exit(1)

# Ensure NLTK resources are available before using them
ensure_nltk_resources()

def preprocess_text(text):
    """
    Preprocess text data by cleaning, tokenizing, removing stopwords, and lemmatizing.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
        
    Raises:
        ValueError: If text is None or empty
        LookupError: If required NLTK resources are not available
    """
    if text is None or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")
    
    if not text.strip():
        return ""
    
    try:
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
    
    except LookupError as e:
        print(f"NLTK resource not found: {str(e)}")
        print("Please ensure all required NLTK resources are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error preprocessing text: {str(e)}")
        return ""

# Apply preprocessing to the text data
preprocessed_texts = texts.apply(preprocess_text)


# In[11]:


preprocessed_texts


# In[12]:


vectorizer = TfidfVectorizer(
    max_df=0.98,  # Ignore words that appear in more than 98% of documents
    min_df=2,     # Ignore words that appear in fewer than 2 documents
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


# Note: Avoiding conversion to dense array to preserve memory efficiency
# For large datasets, working with the sparse matrix X directly is recommended
# Only convert small portions to dense format when necessary for inspection

# Display basic information about the sparse matrix
print(f"TF-IDF Matrix shape: {X.shape}")
print(f"Matrix sparsity: {(1 - X.nnz / (X.shape[0] * X.shape[1])):.4f}")
print(f"Number of non-zero elements: {X.nnz}")

# For demonstration, show a small sample (first 5 rows, first 10 features) as dense
sample_size = min(5, X.shape[0])
feature_sample_size = min(10, X.shape[1])
feature_names = vectorizer.get_feature_names_out()

print(f"\nSample TF-IDF values (first {sample_size} documents, first {feature_sample_size} features):")
sample_df = pd.DataFrame(
    X[:sample_size, :feature_sample_size].toarray(),
    columns=feature_names[:feature_sample_size]
)
print(sample_df)


# In[17]:


# Display shape of the sparse matrix instead of dense DataFrame
print(f"TF-IDF Matrix shape: {X.shape}")


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


from sklearn.decomposition import TruncatedSVD


# In[37]:


# Use TruncatedSVD instead of PCA to handle sparse matrices efficiently
# TruncatedSVD works directly with sparse matrices without densification
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X)


# In[38]:


X_svd


# In[39]:


X_svd.shape


# In[40]:


# Step 4: Plot the data points in 2D SVD space with clusters as colors
plt.figure(figsize=(8, 6))
plt.scatter(X_svd[:, 0], X_svd[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
plt.title("TruncatedSVD of Clusters in 2D Space")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.colorbar(label="Cluster")
plt.show()


# In[ ]:




