#!/usr/bin/env python
# coding: utf-8

"""
Enhanced Chess Endgame Classification
=====================================
Improved version of the King-Rook vs King classification problem with:
- Advanced feature engineering including chess domain knowledge
- Multiple advanced algorithms with hyperparameter optimization
- Comprehensive evaluation metrics and analysis
- Cross-validation and model selection improvements
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold, learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

class ChessEndgameClassifier:
    """Enhanced chess endgame classifier with advanced feature engineering and model selection"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_names = []
        self.scaler = None
        self.label_encoder = None
        
    def load_and_explore_data(self, filepath="king_rook_vs_king.csv"):
        """Load and explore the chess endgame dataset"""
        try:
            self.data = pd.read_csv(filepath)
            print("Dataset loaded successfully!")
            print(f"Shape: {self.data.shape}")
            print(f"\nDataset Info:")
            print(self.data.info())
            print(f"\nTarget distribution:")
            print(self.data['white_depth_of_win'].value_counts())
            print(f"\nFirst few rows:")
            print(self.data.head())
            return self.data
        except FileNotFoundError:
            print(f"File {filepath} not found. Creating sample data...")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample chess endgame data if original file not available"""
        np.random.seed(42)
        n_samples = 5000
        
        # Generate synthetic chess positions
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = list(range(1, 9))
        outcomes = ['draw', 'one', 'two', 'three', 'four', 'five', 'six', 
                   'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 
                   'thirteen', 'fourteen', 'fifteen', 'sixteen']
        
        data = {
            'white_king_file': np.random.choice(files, n_samples),
            'white_king_rank': np.random.choice(ranks, n_samples),
            'white_rook_file': np.random.choice(files, n_samples),
            'white_rook_rank': np.random.choice(ranks, n_samples),
            'black_king_file': np.random.choice(files, n_samples),
            'black_king_rank': np.random.choice(ranks, n_samples),
            'white_depth_of_win': np.random.choice(outcomes, n_samples, 
                                                 p=[0.3] + [0.7/16]*16)  # More draws
        }
        
        self.data = pd.DataFrame(data)
        print("Sample data created!")
        return self.data
    
    def create_advanced_features(self, df):
        """Create advanced features using chess domain knowledge"""
        df_features = df.copy()
        
        # Convert file letters to numbers for calculations
        file_to_num = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
        
        # Original position features
        df_features['white_king_file_num'] = df_features['white_king_file'].map(file_to_num)
        df_features['white_rook_file_num'] = df_features['white_rook_file'].map(file_to_num)
        df_features['black_king_file_num'] = df_features['black_king_file'].map(file_to_num)
        
        # Distance features (chess domain knowledge)
        # Manhattan distance between kings
        df_features['king_manhattan_distance'] = (
            abs(df_features['white_king_file_num'] - df_features['black_king_file_num']) +
            abs(df_features['white_king_rank'] - df_features['black_king_rank'])
        )
        
        # Euclidean distance between kings
        df_features['king_euclidean_distance'] = np.sqrt(
            (df_features['white_king_file_num'] - df_features['black_king_file_num'])**2 +
            (df_features['white_king_rank'] - df_features['black_king_rank'])**2
        )
        
        # Distance from white king to rook
        df_features['king_rook_manhattan'] = (
            abs(df_features['white_king_file_num'] - df_features['white_rook_file_num']) +
            abs(df_features['white_king_rank'] - df_features['white_rook_rank'])
        )
        
        # Distance from black king to rook
        df_features['black_king_rook_manhattan'] = (
            abs(df_features['black_king_file_num'] - df_features['white_rook_file_num']) +
            abs(df_features['black_king_rank'] - df_features['white_rook_rank'])
        )
        
        # Chess-specific features
        # King centralization (distance from center)
        center_files = [4, 5]  # d, e files
        center_ranks = [4, 5]  # 4th, 5th ranks
        
        df_features['white_king_centralization'] = np.minimum(
            abs(df_features['white_king_file_num'] - 4.5),
            abs(df_features['white_king_file_num'] - 4.5)
        ) + np.minimum(
            abs(df_features['white_king_rank'] - 4.5),
            abs(df_features['white_king_rank'] - 4.5)
        )
        
        df_features['black_king_centralization'] = np.minimum(
            abs(df_features['black_king_file_num'] - 4.5),
            abs(df_features['black_king_file_num'] - 4.5)
        ) + np.minimum(
            abs(df_features['black_king_rank'] - 4.5),
            abs(df_features['black_king_rank'] - 4.5)
        )
        
        # Edge proximity (important in king-rook endgames)
        df_features['white_king_edge_distance'] = np.minimum.reduce([
            df_features['white_king_file_num'] - 1,      # Distance to a-file
            8 - df_features['white_king_file_num'],      # Distance to h-file
            df_features['white_king_rank'] - 1,          # Distance to 1st rank
            8 - df_features['white_king_rank']           # Distance to 8th rank
        ])
        
        df_features['black_king_edge_distance'] = np.minimum.reduce([
            df_features['black_king_file_num'] - 1,
            8 - df_features['black_king_file_num'],
            df_features['black_king_rank'] - 1,
            8 - df_features['black_king_rank']
        ])
        
        # Rook mobility simulation (simplified)
        df_features['rook_mobility'] = (
            (8 - df_features['white_rook_file_num']) +  # Files to the right
            (df_features['white_rook_file_num'] - 1) +  # Files to the left
            (8 - df_features['white_rook_rank']) +      # Ranks up
            (df_features['white_rook_rank'] - 1)        # Ranks down
        )
        
        # Relative positions
        df_features['kings_same_rank'] = (
            df_features['white_king_rank'] == df_features['black_king_rank']
        ).astype(int)
        
        df_features['kings_same_file'] = (
            df_features['white_king_file_num'] == df_features['black_king_file_num']
        ).astype(int)
        
        df_features['rook_blocks_king'] = (
            ((df_features['white_rook_file_num'] == df_features['black_king_file_num']) & 
             (df_features['white_rook_rank'] != df_features['black_king_rank'])) |
            ((df_features['white_rook_rank'] == df_features['black_king_rank']) & 
             (df_features['white_rook_file_num'] != df_features['black_king_file_num']))
        ).astype(int)
        
        # One-hot encode categorical features
        categorical_features = ['white_king_file', 'white_rook_file', 'black_king_file']
        df_encoded = pd.get_dummies(df_features, columns=categorical_features, drop_first=True)
        
        return df_encoded
    
    def prepare_data(self):
        """Prepare features and target for modeling"""
        # Create advanced features
        df_features = self.create_advanced_features(self.data)
        
        # Separate features and target
        X = df_features.drop('white_depth_of_win', axis=1)
        y = df_features['white_depth_of_win']
        
        # Create binary classification problem (win vs draw)
        # This simplifies the multiclass problem and is more practical
        y_binary = (y != 'draw').astype(int)
        
        # Also create a simplified multiclass version
        le = LabelEncoder()
        y_multiclass = le.fit_transform(y)
        
        self.feature_names = X.columns.tolist()
        self.label_encoder = le
        
        print(f"Features created: {X.shape[1]}")
        print(f"Binary target distribution: {np.bincount(y_binary)}")
        
        return X, y_binary, y_multiclass
    
    def optimize_models(self, X_train, y_train, cv_folds=5, n_jobs=-1):
        """Optimize multiple models with hyperparameter tuning"""
        
        # Define models with parameter grids
        model_configs = {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=n_jobs),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'poly', 'sigmoid']
                }
            },
            'MLPClassifier': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'learning_rate': ['constant', 'adaptive'],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            }
        }
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        optimized_models = {}
        
        print("Optimizing models...")
        for name, config in model_configs.items():
            print(f"\nOptimizing {name}...")
            
            # Use RandomizedSearchCV for faster optimization
            search = RandomizedSearchCV(
                estimator=config['model'],
                param_distributions=config['params'],
                n_iter=50,  # Reduce iterations for faster execution
                cv=cv,
                scoring='accuracy',
                random_state=42,
                n_jobs=n_jobs,
                verbose=1
            )
            
            search.fit(X_train, y_train)
            optimized_models[name] = {
                'model': search.best_estimator_,
                'best_score': search.best_score_,
                'best_params': search.best_params_
            }
            
            print(f"Best CV Score: {search.best_score_:.4f}")
            print(f"Best Parameters: {search.best_params_}")
        
        self.models = optimized_models
        return optimized_models
    
    def evaluate_models(self, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation"""
        results = {}
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            # ROC AUC for binary classification
            roc_auc = None
            if y_test_proba is not None and len(np.unique(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, y_test_proba)
            
            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_score': model_info['best_score'],
                'overfitting': train_acc - test_acc
            }
            
            print(f"\n{name}:")
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy:  {test_acc:.4f}")
            print(f"  Precision:      {precision:.4f}")
            print(f"  Recall:         {recall:.4f}")
            print(f"  F1 Score:       {f1:.4f}")
            print(f"  ROC AUC:        {roc_auc:.4f}" if roc_auc else "  ROC AUC:        N/A")
            print(f"  CV Score:       {model_info['best_score']:.4f}")
            print(f"  Overfitting:    {train_acc - test_acc:.4f}")
        
        # Find best model based on test accuracy
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        self.best_model = self.models[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
        
        return results
    
    def create_ensemble(self, X_train, y_train):
        """Create an ensemble of the best models"""
        # Select top 3 models based on CV score
        top_models = sorted(self.models.items(), 
                          key=lambda x: x[1]['best_score'], reverse=True)[:3]
        
        voting_models = [(name, info['model']) for name, info in top_models]
        
        ensemble = VotingClassifier(
            estimators=voting_models,
            voting='soft'  # Use predicted probabilities
        )
        
        ensemble.fit(X_train, y_train)
        self.models['Ensemble'] = {
            'model': ensemble,
            'best_score': np.mean([info['best_score'] for _, info in top_models]),
            'best_params': 'Ensemble of top 3 models'
        }
        
        print(f"\n🤝 Ensemble created with models: {[name for name, _ in top_models]}")
        return ensemble
    
    def visualize_results(self, results, X_test, y_test):
        """Create comprehensive visualizations"""
        
        # Model comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model accuracy comparison
        models = list(results.keys())
        test_scores = [results[m]['test_accuracy'] for m in models]
        cv_scores = [results[m]['cv_score'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0,0].bar(x - width/2, test_scores, width, label='Test Accuracy', alpha=0.8)
        axes[0,0].bar(x + width/2, cv_scores, width, label='CV Score', alpha=0.8)
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_title('Model Performance Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(models, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Overfitting analysis
        overfitting = [results[m]['overfitting'] for m in models]
        colors = ['red' if x > 0.05 else 'green' if x < 0.02 else 'orange' for x in overfitting]
        
        axes[0,1].bar(models, overfitting, color=colors, alpha=0.7)
        axes[0,1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='High Overfitting')
        axes[0,1].axhline(y=0.02, color='green', linestyle='--', alpha=0.5, label='Good Generalization')
        axes[0,1].set_xlabel('Models')
        axes[0,1].set_ylabel('Train - Test Accuracy')
        axes[0,1].set_title('Overfitting Analysis')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Metrics comparison
        metrics = ['precision', 'recall', 'f1_score']
        metric_data = {metric: [results[m][metric] for m in models] for metric in metrics}
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            axes[1,0].bar(x + i*width, metric_data[metric], width, 
                         label=metric.replace('_', ' ').title(), alpha=0.8)
        
        axes[1,0].set_xlabel('Models')
        axes[1,0].set_ylabel('Score')
        axes[1,0].set_title('Performance Metrics Comparison')
        axes[1,0].set_xticks(x + width)
        axes[1,0].set_xticklabels(models, rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Feature importance (for the best model)
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            indices = np.argsort(importance)[::-1][:15]  # Top 15 features
            
            axes[1,1].bar(range(len(indices)), importance[indices])
            axes[1,1].set_xlabel('Feature Index')
            axes[1,1].set_ylabel('Importance')
            axes[1,1].set_title('Top 15 Feature Importances (Best Model)')
            axes[1,1].set_xticks(range(len(indices)))
            axes[1,1].set_xticklabels([self.feature_names[i] for i in indices], 
                                     rotation=45, ha='right')
            
        plt.tight_layout()
        plt.show()
        
        # Confusion matrix for best model
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Best Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def run_complete_analysis(self, filepath="king_rook_vs_king.csv"):
        """Run the complete enhanced analysis"""
        print("🚀 Starting Enhanced Chess Endgame Analysis")
        print("=" * 50)
        
        # Load data
        self.load_and_explore_data(filepath)
        
        # Prepare features
        X, y_binary, y_multiclass = self.prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTraining set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        
        # Optimize models
        self.optimize_models(X_train_scaled, y_train)
        
        # Create ensemble
        self.create_ensemble(X_train_scaled, y_train)
        
        # Evaluate models
        results = self.evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Visualizations
        self.visualize_results(results, X_test_scaled, y_test)
        
        print("\n✅ Analysis Complete!")
        print("\nKey Improvements Implemented:")
        print("1. Advanced chess-specific feature engineering")
        print("2. Multiple algorithm comparison with hyperparameter optimization")
        print("3. Comprehensive evaluation metrics (precision, recall, F1, ROC-AUC)")
        print("4. Overfitting analysis and model selection")
        print("5. Ensemble modeling for improved performance")
        print("6. Enhanced visualizations and interpretability")
        print("7. Robust preprocessing and error handling")
        
        return results

# Main execution
if __name__ == "__main__":
    classifier = ChessEndgameClassifier()
    results = classifier.run_complete_analysis()
