"""
Predictive Modeling on Real-World Data
A machine learning pipeline for classification/regression tasks with automated preprocessing,
cross-validation, and performance visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

class PredictiveModel:
    def __init__(self, data_path=None, target_column=None):
        """Initialize the predictive modeling pipeline."""
        self.data_path = data_path
        self.target_column = target_column
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.results = {}
        
    def load_and_clean_data(self):
        """Load data and perform automated cleaning."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Original shape: {self.df.shape}")
        
        # Handle missing values
        print("\nCleaning data...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Impute numeric columns with median
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Impute categorical columns with mode
        for col in categorical_cols:
            if col != self.target_column and self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        # Remove duplicates
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Removed {initial_rows - len(self.df)} duplicate rows")
        
        # Encode categorical variables
        le = LabelEncoder()
        for col in categorical_cols:
            if col != self.target_column:
                self.df[col] = le.fit_transform(self.df[col].astype(str))
        
        print(f"Cleaned shape: {self.df.shape}")
        return self.df
    
    def prepare_features(self):
        """Prepare features and target for modeling."""
        print("\nPreparing features...")
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Encode target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
    def build_pipelines(self):
        """Build ML pipelines with preprocessing."""
        pipelines = {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=1000, random_state=42))
            ]),
            'Random Forest': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'Gradient Boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
            ])
        }
        return pipelines
    
    def train_and_evaluate(self):
        """Train multiple models with cross-validation."""
        print("\nTraining models with cross-validation...")
        pipelines = self.build_pipelines()
        
        for name, pipeline in pipelines.items():
            print(f"\n{name}:")
            
            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, self.X_train, self.y_train, 
                cv=5, scoring='f1_weighted'
            )
            
            # Train on full training set
            pipeline.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = pipeline.predict(self.X_test)
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            self.results[name] = {
                'pipeline': pipeline,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'test_f1': f1,
                'predictions': y_pred
            }
            
            print(f"  CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Test F1 Score: {f1:.4f}")
        
        # Select best model
        best_name = max(self.results, key=lambda x: self.results[x]['test_f1'])
        self.best_model = self.results[best_name]['pipeline']
        print(f"\nBest model: {best_name} (F1: {self.results[best_name]['test_f1']:.4f})")
        
    def visualize_results(self):
        """Create comprehensive performance visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Comparison - F1 Scores
        models = list(self.results.keys())
        f1_scores = [self.results[m]['test_f1'] for m in models]
        cv_scores = [self.results[m]['cv_mean'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, cv_scores, width, label='CV F1', alpha=0.8)
        axes[0, 0].bar(x + width/2, f1_scores, width, label='Test F1', alpha=0.8)
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=15, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Confusion Matrix for best model
        best_name = max(self.results, key=lambda x: self.results[x]['test_f1'])
        y_pred = self.results[best_name]['predictions']
        cm = confusion_matrix(self.y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title(f'Confusion Matrix - {best_name}')
        axes[0, 1].set_ylabel('True Label')
        axes[0, 1].set_xlabel('Predicted Label')
        
        # 3. Cross-validation score distribution
        for name in models:
            cv_scores = self.results[name]['cv_scores']
            axes[1, 0].plot(range(1, len(cv_scores) + 1), cv_scores, 
                          marker='o', label=name, linewidth=2)
        
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Cross-Validation Scores by Fold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Accuracy vs F1 Score scatter
        accuracies = [self.results[m]['test_accuracy'] for m in models]
        axes[1, 1].scatter(accuracies, f1_scores, s=200, alpha=0.6)
        
        for i, name in enumerate(models):
            axes[1, 1].annotate(name, (accuracies[i], f1_scores[i]), 
                              fontsize=9, ha='center')
        
        axes[1, 1].set_xlabel('Accuracy')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Accuracy vs F1 Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'model_performance.png'")
        plt.show()
    
    def generate_report(self):
        """Generate a detailed classification report."""
        best_name = max(self.results, key=lambda x: self.results[x]['test_f1'])
        y_pred = self.results[best_name]['predictions']
        
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION REPORT - {best_name}")
        print(f"{'='*60}")
        print(classification_report(self.y_test, y_pred))
        
    def run_full_pipeline(self):
        """Execute the complete modeling pipeline."""
        self.load_and_clean_data()
        self.prepare_features()
        self.train_and_evaluate()
        self.visualize_results()
        self.generate_report()
        

# Example usage
if __name__ == "__main__":
    # For demonstration with a sample dataset
    # Replace with your actual dataset path
    
    print("Predictive Modeling Pipeline")
    print("=" * 60)
    
    # Create sample data for demonstration
    from sklearn.datasets import make_classification
    
    print("\nGenerating sample dataset (100k records)...")
    X, y = make_classification(
        n_samples=100000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Save sample data
    df_sample = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df_sample['target'] = y
    df_sample.to_csv('sample_data.csv', index=False)
    
    # Run pipeline
    model = PredictiveModel(data_path='sample_data.csv', target_column='target')
    model.run_full_pipeline()
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
