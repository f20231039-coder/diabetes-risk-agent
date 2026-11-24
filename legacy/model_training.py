"""
Model Training Module
Handles training of Naive Bayes and MLP models.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import numpy as np


def train_models(X_train, y_train, use_tuning=False):
    """Train Naive Bayes and MLP models with StandardScaler (best performing)."""
    print(f"\n{'='*80}")
    print("MODEL TRAINING")
    print(f"{'='*80}")
    
    models = {}
    
    if use_tuning:
        print("\n[Mode: Hyperparameter Tuning - This may take a few minutes]")
        
        # Tune Naive Bayes
        print("\n[1/2] Tuning Naive Bayes...")
        nb_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GaussianNB())
        ])
        param_grid_nb = {'classifier__var_smoothing': np.logspace(-10, -8, 10)}
        grid_nb = GridSearchCV(nb_pipeline, param_grid_nb, cv=5, scoring='f1', n_jobs=-1, verbose=0)
        grid_nb.fit(X_train, y_train)
        models['Naive Bayes'] = grid_nb.best_estimator_
        print(f"  [OK] Best params: {grid_nb.best_params_}")
        print(f"  [OK] Best CV F1-score: {grid_nb.best_score_:.4f}")
        
        # Tune MLP
        print("\n[2/2] Tuning MLP...")
        mlp_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(random_state=57, max_iter=1000))
        ])
        param_dist_mlp = {
            'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100)],
            'classifier__activation': ['relu', 'tanh'],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__learning_rate': ['constant', 'adaptive']
        }
        random_mlp = RandomizedSearchCV(mlp_pipeline, param_dist_mlp, n_iter=20, 
                                       cv=5, scoring='f1', n_jobs=-1, verbose=0, random_state=42)
        random_mlp.fit(X_train, y_train)
        models['MLP'] = random_mlp.best_estimator_
        print(f"  [OK] Best params: {random_mlp.best_params_}")
        print(f"  [OK] Best CV F1-score: {random_mlp.best_score_:.4f}")
        
    else:
        print("\n[Mode: Default Parameters - Fast Training]")
        
        # Train Naive Bayes
        print("\n[1/2] Training Naive Bayes...")
        nb_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GaussianNB())
        ])
        nb_pipeline.fit(X_train, y_train)
        models['Naive Bayes'] = nb_pipeline
        train_acc_nb = accuracy_score(y_train, nb_pipeline.predict(X_train))
        print(f"  [OK] Training accuracy: {train_acc_nb:.4f}")
        
        # Train MLP
        print("\n[2/2] Training MLP...")
        mlp_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
        ])
        mlp_pipeline.fit(X_train, y_train)
        models['MLP'] = mlp_pipeline
        train_acc_mlp = accuracy_score(y_train, mlp_pipeline.predict(X_train))
        print(f"  [OK] Training accuracy: {train_acc_mlp:.4f}")
    
    print(f"\n[OK] Model training complete")
    return models
