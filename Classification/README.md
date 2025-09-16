# Scholar Text Classification

This project performs text classification and clustering analysis on scholarly activities data.

## Recent Improvements

The following issues have been fixed to improve robustness, memory efficiency, and maintainability:

### High Priority Fixes
- **Sparse Matrix Handling**: Replaced PCA with TruncatedSVD to avoid memory-intensive densification of sparse TF-IDF matrices

### Medium Priority Fixes
- **Error Handling**: Added comprehensive error handling for Excel file reading with meaningful error messages
- **NLTK Resources**: Added automatic checking and downloading of required NLTK resources (punkt, stopwords, wordnet)
- **Memory Optimization**: Avoided converting large sparse matrices to dense DataFrames
- **Unit Tests**: Added comprehensive unit tests for the text preprocessing function

### Low Priority Fixes
- **Configuration**: Replaced hardcoded file path with configurable environment variable
- **Documentation**: Fixed misaligned comments in TfidfVectorizer parameters

## Usage

### Setting up the data file
Instead of using a hardcoded path, you can now:
1. Place your `Score.xlsx` file in the same directory as the script, or
2. Set the `SCHOLAR_DATA_FILE` environment variable:
   ```bash
   export SCHOLAR_DATA_FILE="/path/to/your/Score.xlsx"
   ```

### Running the main script
```bash
python Scholar.py
```

### Running the tests
```bash
python test_scholar.py
```

## Dependencies

- pandas
- scikit-learn
- nltk
- matplotlib
- numpy

## Features

1. **Text Preprocessing**: Robust text cleaning, tokenization, stopword removal, and lemmatization
2. **TF-IDF Vectorization**: Efficient sparse matrix representation
3. **K-Means Clustering**: Document clustering with silhouette score evaluation
4. **Dimensionality Reduction**: TruncatedSVD for efficient sparse matrix dimensionality reduction
5. **Visualization**: 2D visualization of clusters
6. **Error Handling**: Comprehensive error handling throughout the pipeline

## Memory Efficiency

The code now handles large datasets more efficiently by:
- Using TruncatedSVD instead of PCA to avoid densifying sparse matrices
- Avoiding conversion of sparse TF-IDF matrices to dense DataFrames
- Working with sparse matrices throughout the pipeline where possible
