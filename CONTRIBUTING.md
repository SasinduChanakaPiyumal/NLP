# Contributing to Parliamentary Debates Text Classification

First off, thank you for considering contributing to this project! 🎉 

The following is a set of guidelines for contributing to the Parliamentary Debates Text Classification project. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## 🚀 How Can I Contribute?

### Reporting Bugs 🐛

Before creating bug reports, please check the existing issues as you might find that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed after following the steps**
- **Explain which behavior you expected to see instead and why**
- **Include screenshots and animated GIFs if helpful**
- **Include your environment details** (Python version, OS, package versions)

### Suggesting Enhancements 💡

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior** and **explain which behavior you expected to see instead**
- **Explain why this enhancement would be useful**
- **List some other text analysis or ML projects where this enhancement exists**

### Your First Code Contribution 🔰

Unsure where to begin contributing? You can start by looking through these issue labels:

- `good-first-issue` - issues that should only require a few lines of code
- `help-wanted` - issues that should be a bit more involved than beginner issues

### Areas for Contribution

1. **Algorithm Improvements**
   - Implement new clustering algorithms (DBSCAN, Hierarchical clustering)
   - Add topic modeling (LDA, NMF)
   - Improve text preprocessing pipeline

2. **Data Analysis**
   - Add more comprehensive evaluation metrics
   - Implement cross-validation for clustering
   - Add statistical analysis of results

3. **Visualization**
   - Interactive visualizations using Plotly
   - Better cluster visualization techniques
   - Word clouds for cluster analysis

4. **Code Quality**
   - Add unit tests for core functions
   - Improve error handling
   - Add type hints throughout the codebase

5. **Documentation**
   - Improve inline code documentation
   - Add more examples and tutorials
   - Create API documentation

## 🛠️ Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/parliamentary-debates-classification.git
   cd parliamentary-debates-classification
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

3. **Download Required NLTK Data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

4. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line Length**: Maximum 88 characters (Black default)
- **Quotes**: Use double quotes for strings
- **Imports**: Group imports in the following order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library specific imports

### Code Formatting

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
# Install Black
pip install black

# Format all Python files
black .

# Check formatting without making changes
black --check .
```

### Linting

We use [flake8](https://flake8.pycqa.org/) for linting:

```bash
# Install flake8
pip install flake8

# Run linting
flake8 .
```

### Type Hints

Please add type hints to new functions:

```python
from typing import List, Dict, Optional

def preprocess_text(text: str) -> str:
    """Preprocess text for analysis."""
    # Implementation here
    return processed_text

def cluster_texts(texts: List[str], n_clusters: int) -> Dict[str, List[int]]:
    """Cluster texts using K-means."""
    # Implementation here
    return {"labels": labels, "centers": centers}
```

### Documentation

- Use Google-style docstrings for all functions and classes
- Include examples in docstrings where appropriate
- Keep comments concise but informative

Example docstring:
```python
def calculate_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate silhouette score for clustering results.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        labels: Cluster labels for each sample
        
    Returns:
        Silhouette score between -1 and 1, where higher is better
        
    Example:
        >>> X = [[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]
        >>> labels = [0, 0, 0, 1, 1, 1]
        >>> score = calculate_silhouette_score(X, labels)
        >>> print(f"Silhouette score: {score:.3f}")
    """
```

## 📤 Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Type
- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools

#### Examples
```
feat(clustering): add DBSCAN clustering algorithm

Add DBSCAN as an alternative to K-means clustering with automatic
cluster number detection.

Closes #123
```

```
fix(preprocessing): handle empty strings in text processing

- Add validation for empty/null strings
- Return empty string instead of raising exception
- Add unit tests for edge cases

Fixes #456
```

## 🔄 Pull Request Process

1. **Update Documentation**
   - Update the README.md with details of changes if applicable
   - Update docstrings and inline comments

2. **Add Tests**
   - Add tests for new functionality
   - Ensure all tests pass
   - Maintain or improve test coverage

3. **Follow Coding Standards**
   - Run Black formatting: `black .`
   - Run linting: `flake8 .`
   - Fix any issues found

4. **Create Pull Request**
   - Use a clear and descriptive title
   - Fill out the pull request template
   - Reference any related issues

5. **Review Process**
   - Pull requests require at least one review
   - Address feedback and requested changes
   - Maintain a respectful and collaborative tone

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
```

## 🐞 Issue Reporting

### Bug Reports

Use the following template for bug reports:

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
- OS: [e.g. Windows 10, macOS 11.2, Ubuntu 20.04]
- Python Version: [e.g. 3.9.1]
- Package versions: [run `pip list` and paste relevant packages]

**Additional context**
Add any other context about the problem here.
```

### Feature Requests

Use the following template for feature requests:

```markdown
**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

## 🏷️ Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature or request  
- `good-first-issue` - Good for newcomers
- `help-wanted` - Extra attention is needed
- `documentation` - Improvements or additions to documentation
- `question` - Further information is requested
- `wontfix` - This will not be worked on
- `duplicate` - This issue or pull request already exists

## 🎉 Recognition

Contributors will be recognized in our README.md file and release notes. We appreciate all contributions, no matter how small!

## 📞 Questions?

Don't hesitate to ask questions! You can:
- Open an issue with the `question` label
- Contact the maintainers directly
- Start a discussion in the repository

Thank you for contributing! 🚀
