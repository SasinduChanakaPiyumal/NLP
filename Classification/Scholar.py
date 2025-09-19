#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# ## Loading and preprocessing data

def load_data(file_path):
    """Load data with error handling for different file formats"""
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Load data (update path as needed)
try:
    df = pd.read_excel('C:\\Users\\Maclan\\OneDrive\\Desktop\\Score.xlsx')
    print("Data shape:", df.shape)
    print("Null values:", df.isnull().sum().sum())
    print("Duplicates in ACTIVITIES:", df['ACTIVITIES'].duplicated().sum() if 'ACTIVITIES' in df.columns else "Column not found")
    
    # Clean data
    df = df.dropna().drop_duplicates()
    texts = df['ACTIVITIES'] if 'ACTIVITIES' in df.columns else df.iloc[:, 0]  # Use first column if ACTIVITIES not found
    print(f"Final data shape after cleaning: {df.shape}")
    print(f"Number of text samples: {len(texts)}")
    
except Exception as e:
    print(f"Error with original file: {e}")
    print("Creating sample data for demonstration...")
    # Create sample data for demonstration if original file not available
    sample_texts = [
        "research development artificial intelligence machine learning",
        "education teaching students university academic research",
        "healthcare medical treatment patient clinical study",
        "technology innovation software development programming",
        "business management marketing finance economics",
        "environment climate change sustainability green energy",
        "politics government policy public administration",
        "arts culture literature creative writing music"
    ] * 20  # Repeat to create more samples
    texts = pd.Series(sample_texts)
    print(f"Using sample data with {len(texts)} text samples")


## Enhanced Text Preprocessing

def advanced_preprocess_text(text, min_word_length=2, max_word_length=20):
    """
    Enhanced text preprocessing with domain-specific improvements
    """
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs, email addresses, and special patterns
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)  # Keep only alphanumeric and spaces
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    words = nltk.word_tokenize(text)
    
    # Enhanced stopword removal (including custom academic stopwords)
    stop_words = set(stopwords.words('english'))
    # Add common academic/research stopwords
    academic_stopwords = {'also', 'would', 'could', 'may', 'might', 'use', 'using', 'used', 
                         'study', 'studies', 'research', 'paper', 'article', 'journal', 
                         'university', 'college', 'department', 'student', 'students'}
    stop_words.update(academic_stopwords)
    
    # Filter words by length and remove stopwords
    words = [word for word in words if len(word) >= min_word_length and 
             len(word) <= max_word_length and word not in stop_words and word.isalpha()]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Apply enhanced preprocessing
print("Preprocessing text data...")
preprocessed_texts = texts.apply(advanced_preprocess_text)
print(f"Preprocessing completed. Non-empty texts: {sum(len(text) > 0 for text in preprocessed_texts)}")

# Filter out empty texts
valid_indices = preprocessed_texts.str.len() > 0
preprocessed_texts = preprocessed_texts[valid_indices]
texts = texts[valid_indices]
print(f"After filtering empty texts: {len(preprocessed_texts)} samples")


## Enhanced Feature Engineering

def create_optimized_features(texts, max_features=5000):
    """
    Create optimized TF-IDF features with hyperparameter tuning
    """
    # Enhanced TF-IDF with better parameters
    vectorizer = TfidfVectorizer(
        max_df=0.85,  # Ignore words appearing in more than 85% of documents
        min_df=3,     # Ignore words appearing in fewer than 3 documents
        max_features=max_features,  # Limit vocabulary size
        ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
        stop_words='english',
        sublinear_tf=True,  # Apply sublinear scaling
        norm='l2',  # L2 normalization
        use_idf=True,
        smooth_idf=True,
        token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ letters
    )
    
    # Transform texts
    X_tfidf = vectorizer.fit_transform(texts)
    
    # Apply dimensionality reduction for better clustering
    # Use TruncatedSVD for sparse matrices
    n_components = min(100, X_tfidf.shape[1] - 1)  # Ensure components < features
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)
    
    print(f"Original TF-IDF shape: {X_tfidf.shape}")
    print(f"Reduced features shape: {X_reduced.shape}")
    print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
    print(f"Top 10 features: {vectorizer.get_feature_names_out()[:10]}")
    
    return X_tfidf, X_reduced, vectorizer, svd

# Create optimized features
X_tfidf, X_features, vectorizer, svd = create_optimized_features(preprocessed_texts)

# Display feature information
print("\n=== Feature Engineering Results ===")
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
print(f"Feature matrix sparsity: {(1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])):.3f}")


## Enhanced Clustering with Multiple Algorithms

def find_optimal_clusters(X, max_k=15):
    """
    Find optimal number of clusters using multiple metrics
    """
    k_range = range(2, min(max_k + 1, X.shape[0]))  # Ensure k < n_samples
    
    metrics = {
        'inertia': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': []
    }
    
    for k in k_range:
        # KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        metrics['inertia'].append(kmeans.inertia_)
        metrics['silhouette'].append(silhouette_score(X, labels))
        metrics['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
        metrics['davies_bouldin'].append(davies_bouldin_score(X, labels))
    
    # Find optimal k using elbow method for inertia
    inertia_diff = np.diff(metrics['inertia'])
    elbow_point = np.argmax(np.diff(inertia_diff)) + 2  # +2 because of double diff
    elbow_point = max(2, min(elbow_point, max_k))
    
    # Find optimal k using silhouette score
    sil_optimal = k_range[np.argmax(metrics['silhouette'])]
    
    # Plot evaluation metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Clustering Evaluation Metrics', fontsize=16)
    
    # Inertia (Elbow Method)
    axes[0,0].plot(k_range, metrics['inertia'], marker='o')
    axes[0,0].axvline(x=elbow_point, color='r', linestyle='--', label=f'Elbow: k={elbow_point}')
    axes[0,0].set_title('Elbow Method (Inertia)')
    axes[0,0].set_xlabel('Number of Clusters')
    axes[0,0].set_ylabel('Inertia')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Silhouette Score
    axes[0,1].plot(k_range, metrics['silhouette'], marker='o', color='green')
    axes[0,1].axvline(x=sil_optimal, color='r', linestyle='--', label=f'Max: k={sil_optimal}')
    axes[0,1].set_title('Silhouette Score')
    axes[0,1].set_xlabel('Number of Clusters')
    axes[0,1].set_ylabel('Silhouette Score')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Calinski-Harabasz Score
    ch_optimal = k_range[np.argmax(metrics['calinski_harabasz'])]
    axes[1,0].plot(k_range, metrics['calinski_harabasz'], marker='o', color='orange')
    axes[1,0].axvline(x=ch_optimal, color='r', linestyle='--', label=f'Max: k={ch_optimal}')
    axes[1,0].set_title('Calinski-Harabasz Score')
    axes[1,0].set_xlabel('Number of Clusters')
    axes[1,0].set_ylabel('CH Score')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Davies-Bouldin Score (lower is better)
    db_optimal = k_range[np.argmin(metrics['davies_bouldin'])]
    axes[1,1].plot(k_range, metrics['davies_bouldin'], marker='o', color='red')
    axes[1,1].axvline(x=db_optimal, color='r', linestyle='--', label=f'Min: k={db_optimal}')
    axes[1,1].set_title('Davies-Bouldin Score')
    axes[1,1].set_xlabel('Number of Clusters')
    axes[1,1].set_ylabel('DB Score')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Optimal clusters - Elbow: {elbow_point}, Silhouette: {sil_optimal}")
    print(f"Optimal clusters - Calinski-Harabasz: {ch_optimal}, Davies-Bouldin: {db_optimal}")
    
    # Return the most commonly suggested optimal k
    suggested_k = max(set([elbow_point, sil_optimal, ch_optimal, db_optimal]), 
                     key=[elbow_point, sil_optimal, ch_optimal, db_optimal].count)
    
    return suggested_k, metrics

def compare_clustering_algorithms(X, n_clusters=5):
    """
    Compare different clustering algorithms
    """
    algorithms = {
        'KMeans': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        'KMeans++': KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10),
        'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
    }
    
    # Add DBSCAN with automatic cluster detection
    # Estimate eps using k-distance
    from sklearn.neighbors import NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=4)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    eps = np.percentile(distances, 90)  # Use 90th percentile as eps
    
    algorithms['DBSCAN'] = DBSCAN(eps=eps, min_samples=3)
    
    results = {}
    
    for name, algorithm in algorithms.items():
        try:
            labels = algorithm.fit_predict(X)
            
            # Skip if only one cluster or too many clusters
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters_found < 2:
                continue
                
            # Calculate metrics (skip silhouette if only one cluster)
            if len(set(labels)) > 1:
                sil_score = silhouette_score(X, labels)
                ch_score = calinski_harabasz_score(X, labels)
                db_score = davies_bouldin_score(X, labels)
                
                results[name] = {
                    'labels': labels,
                    'n_clusters': n_clusters_found,
                    'silhouette': sil_score,
                    'calinski_harabasz': ch_score,
                    'davies_bouldin': db_score,
                    'algorithm': algorithm
                }
        except Exception as e:
            print(f"Error with {name}: {e}")
            continue
    
    # Print comparison
    print("\n=== Clustering Algorithm Comparison ===")
    print(f"{'Algorithm':<15} {'Clusters':<10} {'Silhouette':<12} {'CH Score':<12} {'DB Score':<12}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{name:<15} {result['n_clusters']:<10} {result['silhouette']:<12.3f} "
              f"{result['calinski_harabasz']:<12.1f} {result['davies_bouldin']:<12.3f}")
    
    # Find best algorithm based on silhouette score
    if results:
        best_algorithm = max(results.keys(), key=lambda k: results[k]['silhouette'])
        print(f"\nBest algorithm: {best_algorithm} (Silhouette: {results[best_algorithm]['silhouette']:.3f})")
        return results[best_algorithm], results
    else:
        print("No valid clustering results found")
        return None, {}

# Find optimal number of clusters
print("Finding optimal number of clusters...")
optimal_k, cluster_metrics = find_optimal_clusters(X_features)

# Compare clustering algorithms with optimal k
print(f"\nComparing clustering algorithms with k={optimal_k}...")
best_result, all_results = compare_clustering_algorithms(X_features, optimal_k)

if best_result:
    best_labels = best_result['labels']
    print(f"\nUsing best algorithm with {best_result['n_clusters']} clusters")
    print(f"Final Silhouette Score: {best_result['silhouette']:.3f}")
else:
    # Fallback to simple KMeans
    print("Using fallback KMeans clustering...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    best_labels = kmeans.fit_predict(X_features)
    print(f"Fallback Silhouette Score: {silhouette_score(X_features, best_labels):.3f}")


# ## Hyperparameter Optimization

def optimize_kmeans_hyperparameters(X, k_range=None):
    """
    Optimize KMeans hyperparameters using grid search
    """
    if k_range is None:
        k_range = range(2, min(11, X.shape[0]))
    
    param_grid = {
        'n_clusters': k_range,
        'init': ['k-means++', 'random'],
        'n_init': [10, 20, 30],
        'max_iter': [200, 300, 500]
    }
    
    best_score = -1
    best_params = None
    best_labels = None
    
    print("Optimizing KMeans hyperparameters...")
    total_combinations = len(ParameterGrid(param_grid))
    current = 0
    
    for params in ParameterGrid(param_grid):
        current += 1
        if current % 10 == 0:
            print(f"Progress: {current}/{total_combinations}")
            
        try:
            kmeans = KMeans(**params, random_state=42)
            labels = kmeans.fit_predict(X)
            
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_labels = labels
        except Exception as e:
            continue
    
    print(f"Best parameters: {best_params}")
    print(f"Best silhouette score: {best_score:.3f}")
    
    return best_params, best_labels, best_score

def analyze_cluster_characteristics(X_tfidf, labels, vectorizer, top_n=10):
    """
    Analyze and display cluster characteristics
    """
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)
    
    print("\n=== Cluster Analysis ===")
    
    # Convert to dense array for analysis
    X_dense = X_tfidf.toarray()
    feature_names = vectorizer.get_feature_names_out()
    
    cluster_info = []
    
    for cluster_id in unique_labels:
        if cluster_id == -1:  # Skip noise cluster in DBSCAN
            continue
            
        cluster_mask = labels == cluster_id
        cluster_size = np.sum(cluster_mask)
        
        if cluster_size == 0:
            continue
            
        # Calculate cluster centroid
        cluster_center = np.mean(X_dense[cluster_mask], axis=0)
        
        # Get top features for this cluster
        top_indices = cluster_center.argsort()[-top_n:][::-1]
        top_features = [(feature_names[i], cluster_center[i]) for i in top_indices]
        
        cluster_info.append({
            'id': cluster_id,
            'size': cluster_size,
            'percentage': (cluster_size / len(labels)) * 100,
            'top_features': top_features
        })
        
        print(f"\nCluster {cluster_id + 1}:")
        print(f"  Size: {cluster_size} documents ({cluster_size/len(labels)*100:.1f}%)")
        print(f"  Top {top_n} features:")
        for feature, score in top_features:
            print(f"    {feature}: {score:.3f}")
    
    return cluster_info

def create_enhanced_visualizations(X, X_reduced, labels, title_prefix=""):
    """
    Create comprehensive visualizations of clustering results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title_prefix} Clustering Results', fontsize=16)
    
    # 1. PCA Visualization (2D)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_reduced)
    
    scatter = axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    axes[0,0].set_title('PCA Visualization (2D)')
    axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=axes[0,0])
    
    # 2. t-SNE Visualization (if available)
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_reduced)-1))
        X_tsne = tsne.fit_transform(X_reduced)
        
        scatter = axes[0,1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
        axes[0,1].set_title('t-SNE Visualization (2D)')
        axes[0,1].set_xlabel('t-SNE 1')
        axes[0,1].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[0,1])
    except ImportError:
        axes[0,1].text(0.5, 0.5, 't-SNE not available', ha='center', va='center')
        axes[0,1].set_title('t-SNE Visualization (Not Available)')
    
    # 3. Cluster size distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    axes[1,0].bar(unique_labels, counts)
    axes[1,0].set_title('Cluster Size Distribution')
    axes[1,0].set_xlabel('Cluster ID')
    axes[1,0].set_ylabel('Number of Documents')
    
    # 4. Silhouette analysis
    if len(set(labels)) > 1:
        from sklearn.metrics import silhouette_samples
        silhouette_scores = silhouette_samples(X_reduced, labels)
        
        y_lower = 10
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:  # Skip noise cluster
                continue
                
            cluster_silhouette_scores = silhouette_scores[labels == cluster_id]
            cluster_silhouette_scores.sort()
            
            size_cluster_i = cluster_silhouette_scores.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.nipy_spectral(float(cluster_id) / len(set(labels)))
            axes[1,1].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_scores,
                                   facecolor=color, edgecolor=color, alpha=0.7)
            
            axes[1,1].text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_id))
            y_lower = y_upper + 10
        
        axes[1,1].axvline(x=silhouette_score(X_reduced, labels), color="red", linestyle="--")
        axes[1,1].set_title('Silhouette Analysis')
        axes[1,1].set_xlabel('Silhouette Score')
        axes[1,1].set_ylabel('Cluster')
    else:
        axes[1,1].text(0.5, 0.5, 'Silhouette analysis\nnot available', ha='center', va='center')
        axes[1,1].set_title('Silhouette Analysis (Not Available)')
    
    plt.tight_layout()
    plt.show()

# Perform hyperparameter optimization
print("=== Hyperparameter Optimization ===")
best_params, optimized_labels, optimized_score = optimize_kmeans_hyperparameters(X_features)

# Analyze cluster characteristics using original TF-IDF features
cluster_info = analyze_cluster_characteristics(X_tfidf, optimized_labels, vectorizer)

# Create enhanced visualizations
create_enhanced_visualizations(X_tfidf, X_features, optimized_labels, "Optimized KMeans")

# Final model evaluation
print("\n=== Final Model Evaluation ===")
print(f"Number of clusters: {len(set(optimized_labels))}")
print(f"Silhouette Score: {optimized_score:.3f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz_score(X_features, optimized_labels):.1f}")
print(f"Davies-Bouldin Score: {davies_bouldin_score(X_features, optimized_labels):.3f}")

# Save results summary
results_summary = {
    'optimal_clusters': len(set(optimized_labels)),
    'best_params': best_params,
    'silhouette_score': optimized_score,
    'cluster_sizes': {i: sum(optimized_labels == i) for i in set(optimized_labels)},
    'feature_engineering': {
        'original_features': X_tfidf.shape[1],
        'reduced_features': X_features.shape[1],
        'explained_variance': svd.explained_variance_ratio_.sum()
    }
}

print("\n=== Results Summary ===")
for key, value in results_summary.items():
    print(f"{key}: {value}")

print("\n=== Model Performance Analysis Complete ===")
print("Key improvements implemented:")
print("1. Enhanced text preprocessing with domain-specific stopwords")
print("2. Optimized TF-IDF parameters with trigrams and sublinear scaling")
print("3. Dimensionality reduction using TruncatedSVD")
print("4. Multiple clustering algorithm comparison")
print("5. Comprehensive hyperparameter optimization")
print("6. Multi-metric evaluation (Silhouette, Calinski-Harabasz, Davies-Bouldin)")
print("7. Enhanced visualizations with PCA and t-SNE")
print("8. Detailed cluster analysis and interpretation")




