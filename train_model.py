"""
Fertilizer Recommendation System - ML Pipeline
================================================
A complete machine learning pipeline for fertilizer recommendations based on soil parameters.

This module:
- Loads and combines multiple agricultural datasets
- Performs comprehensive data preprocessing
- Trains multiple classification models
- Evaluates model performance
- Saves trained models for prediction
- Provides prediction functionality

Required CSV Files:
- soil_health_card.csv
- maharashtra_soil_data.csv
- fertilizer_recommendation.csv

Author: ML Pipeline
Date: 2026
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# Try importing XGBoost, fall back if not available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_datasets(data_dir='.'):
    """
    Load all required CSV datasets.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing CSV files (default: current directory)
    
    Returns:
    --------
    dict : Dictionary containing loaded dataframes
    """
    print("=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)
    
    datasets = {}
    file_paths = {
        'soil_health': 'soil_health_card.csv',
        'maharashtra': 'maharashtra_soil_data.csv',
        'fertilizer': 'fertilizer_recommendation.csv'
    }
    
    for name, filename in file_paths.items():
        filepath = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(filepath)
            datasets[name] = df
            print(f"✓ Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        except FileNotFoundError:
            print(f"✗ Warning: {filename} not found at {filepath}")
        except Exception as e:
            print(f"✗ Error loading {filename}: {str(e)}")
    
    if not datasets:
        raise FileNotFoundError(
            "No datasets found. Please ensure CSV files are in the working directory."
        )
    
    return datasets


# ============================================================================
# DATA PREPROCESSING FUNCTIONS
# ============================================================================

def standardize_columns(df):
    """
    Standardize column names to uppercase for consistency.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame : Dataframe with standardized column names
    """
    df.columns = df.columns.str.upper().str.strip()
    return df


def preprocess_dataset(df, soil_columns=['N', 'P', 'K', 'PH'], 
                       crop_column='CROP', fertilizer_column='FERTILIZER'):
    """
    Preprocess individual dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    soil_columns : list
        Names of soil nutrient columns
    crop_column : str
        Name of crop column
    fertilizer_column : str
        Name of fertilizer column
    
    Returns:
    --------
    pd.DataFrame : Preprocessed dataframe with required columns
    """
    df = standardize_columns(df)
    
    # Rename common alternative column names
    rename_map = {
        'NITROGEN': 'N',
        'PHOSPHORUS': 'P',
        'POTASSIUM': 'K',
        'PH_LEVEL': 'PH',
        'PH_': 'PH',
        'FERTILISER': 'FERTILIZER',
        'FERTILISER_NAME': 'FERTILIZER',
        'FERT_NAME': 'FERTILIZER',
        'CROP_NAME': 'CROP'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Check for required columns
    available_columns = []
    for col in soil_columns + [crop_column]:
        if col in df.columns:
            available_columns.append(col)
    
    # Add fertilizer column if available
    if fertilizer_column in df.columns:
        available_columns.append(fertilizer_column)
    
    # If fertilizer not available, we may need to create it from other columns
    if fertilizer_column not in df.columns:
        # Check for alternative columns that might help
        fertilizer_cols = [c for c in df.columns if 'FERTILIZER' in c or 'FERT' in c]
        if fertilizer_cols:
            available_columns.append(fertilizer_cols[0])
    
    # Select only available columns
    if available_columns:
        df = df[available_columns].copy()
    
    # Handle missing values
    # For soil columns: fill with median
    for col in soil_columns:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical columns: fill with mode
    if crop_column in df.columns:
        df[crop_column].fillna(df[crop_column].mode()[0] if not df[crop_column].mode().empty else 'Unknown', inplace=True)
    
    if fertilizer_column in df.columns:
        df[fertilizer_column].fillna(
            df[fertilizer_column].mode()[0] if not df[fertilizer_column].mode().empty else 'Unknown',
            inplace=True
        )
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Remove rows with any missing values in required columns
    required_cols = soil_columns + [crop_column, fertilizer_column]
    df = df.dropna(subset=[col for col in required_cols if col in df.columns])
    
    return df


# ============================================================================
# DATA MERGING FUNCTIONS
# ============================================================================

def merge_datasets(datasets):
    """
    Merge multiple datasets into a single dataframe.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary of loaded dataframes
    
    Returns:
    --------
    pd.DataFrame : Merged dataframe with all data
    """
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING AND MERGING")
    print("=" * 70)
    
    processed_datasets = []
    
    for name, df in datasets.items():
        print(f"\nProcessing {name} dataset...")
        print(f"  Original shape: {df.shape}")
        
        # Preprocess each dataset
        df = preprocess_dataset(df)
        print(f"  After preprocessing: {df.shape}")
        
        if not df.empty:
            processed_datasets.append(df)
    
    if not processed_datasets:
        raise ValueError("No valid data after preprocessing")
    
    # Concatenate all datasets
    merged_df = pd.concat(processed_datasets, axis=0, ignore_index=True)
    
    # Remove any duplicate rows
    merged_df.drop_duplicates(inplace=True)
    
    print(f"\n✓ Merged dataset shape: {merged_df.shape}")
    print(f"✓ Final columns: {list(merged_df.columns)}")
    
    return merged_df


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def normalize_numerical_features(df, numerical_cols=['N', 'P', 'K', 'PH']):
    """
    Normalize numerical features to 0-100 range for consistency.
    Uses RobustScaler for better outlier handling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numerical_cols : list
        Names of numerical columns to normalize
    
    Returns:
    --------
    pd.DataFrame : Dataframe with normalized features
    """
    df = df.copy()
    
    # Use RobustScaler for better outlier handling
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    
    for col in numerical_cols:
        if col in df.columns:
            # RobustScaler for outlier resistance
            df[col] = scaler.fit_transform(df[[col]])
            # Normalize to 0-100 range
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df[col] = ((df[col] - min_val) / (max_val - min_val)) * 100
            else:
                df[col] = 50.0  # Default to median if no variance
    
    return df


def create_advanced_features(df, numerical_cols=['N', 'P', 'K', 'PH']):
    """
    Create advanced feature engineering for improved accuracy.
    Includes polynomial features, interactions, and domain-specific features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with normalized features
    numerical_cols : list
        Names of numerical columns
    
    Returns:
    --------
    pd.DataFrame : Dataframe with engineered features
    """
    df = df.copy()
    
    if all(col in df.columns for col in ['N', 'P', 'K']):
        # NPK Ratios
        df['N_P_Ratio'] = df['N'] / (df['P'] + 1)
        df['N_K_Ratio'] = df['N'] / (df['K'] + 1)
        df['P_K_Ratio'] = df['P'] / (df['K'] + 1)
        df['K_P_Ratio'] = df['K'] / (df['P'] + 1)
        
        # NPK Sum and Balance
        df['NPK_Sum'] = df['N'] + df['P'] + df['K']
        df['NPK_Balance'] = df[['N', 'P', 'K']].std(axis=1)
        df['NPK_Mean'] = df[['N', 'P', 'K']].mean(axis=1)
        
        # NPK Product (interaction term)
        df['N_P_Product'] = df['N'] * df['P']
        df['N_K_Product'] = df['N'] * df['K']
        df['P_K_Product'] = df['P'] * df['K']
        df['NPK_Product'] = df['N'] * df['P'] * df['K']
        
        # Polynomial features (degree 2)
        df['N_Squared'] = df['N'] ** 2
        df['P_Squared'] = df['P'] ** 2
        df['K_Squared'] = df['K'] ** 2
        
        # Dominant nutrient (one-hot encoded concept)
        df['N_Dominant'] = (df['N'] > df['P']) & (df['N'] > df['K'])
        df['P_Dominant'] = (df['P'] > df['N']) & (df['P'] > df['K'])
        df['K_Dominant'] = (df['K'] > df['N']) & (df['K'] > df['P'])
        
        # Nutrient deficiency and excess
        df['High_N'] = (df['N'] > df['N'].quantile(0.75)).astype(int)
        df['High_P'] = (df['P'] > df['P'].quantile(0.75)).astype(int)
        df['High_K'] = (df['K'] > df['K'].quantile(0.75)).astype(int)
        df['Low_N'] = (df['N'] < df['N'].quantile(0.25)).astype(int)
        df['Low_P'] = (df['P'] < df['P'].quantile(0.25)).astype(int)
        df['Low_K'] = (df['K'] < df['K'].quantile(0.25)).astype(int)
    
    # pH-based features (enhanced)
    if 'PH' in df.columns:
        # pH level categorization
        ph_categories = pd.cut(df['PH'], 
                               bins=[0, 5.5, 6.5, 7.5, 8.5, 14],
                               labels=['Very_Acidic', 'Acidic', 'Neutral', 'Alkaline', 'Very_Alkaline'])
        df['pH_Level'] = ph_categories.cat.codes.fillna(ph_categories.cat.codes.median())
        
        # pH-based polynomial features
        df['PH_Squared'] = df['PH'] ** 2
        
        # Distance from neutral pH
        df['pH_Distance_From_Neutral'] = abs(df['PH'] - 7.0)
        
        # pH-Nutrient interactions
        if 'N' in df.columns:
            df['PH_N_Interaction'] = df['PH'] * df['N']
            df['PH_P_Interaction'] = df['PH'] * df['P']
            df['PH_K_Interaction'] = df['PH'] * df['K']
    
    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    return df


def encode_categorical_features(df, crop_column='CROP', fertilizer_column='FERTILIZER',
                               crop_encoder=None, fertilizer_encoder=None):
    """
    Encode categorical features using LabelEncoder.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    crop_column : str
        Name of crop column
    fertilizer_column : str
        Name of fertilizer column
    crop_encoder : LabelEncoder or None
        Existing encoder for crop (if training mode, pass None)
    fertilizer_encoder : LabelEncoder or None
        Existing encoder for fertilizer (if training mode, pass None)
    
    Returns:
    --------
    tuple : (dataframe, crop_encoder, fertilizer_encoder)
    """
    df = df.copy()
    
    # Encode Crop column
    if crop_encoder is None:
        crop_encoder = LabelEncoder()
        df[crop_column] = crop_encoder.fit_transform(df[crop_column].astype(str))
    else:
        df[crop_column] = crop_encoder.transform(df[crop_column].astype(str))
    
    # Encode Fertilizer column
    if fertilizer_encoder is None:
        fertilizer_encoder = LabelEncoder()
        df[fertilizer_column] = fertilizer_encoder.fit_transform(df[fertilizer_column].astype(str))
    else:
        df[fertilizer_column] = fertilizer_encoder.transform(df[fertilizer_column].astype(str))
    
    return df, crop_encoder, fertilizer_encoder


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

# ============================================================================
# HYPERPARAMETER TUNING FUNCTIONS
# ============================================================================

def tune_random_forest(X_train, y_train):
    """
    Tune Random Forest hyperparameters using RandomizedSearchCV.
    Balanced search space for efficiency and accuracy.
    
    Parameters:
    -----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    y_train : np.ndarray or pd.Series
        Training labels
    
    Returns:
    --------
    RandomForestClassifier : Best model found
    """
    print("\n  Tuning Random Forest hyperparameters...")
    
    param_dist = {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced_subsample')
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(
        rf,
        param_dist,
        n_iter=15,
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"  Best params: {random_search.best_params_}")
    print(f"  Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


def tune_svm(X_train, y_train):
    """
    Tune SVM hyperparameters using RandomizedSearchCV with expanded search space.
    Enhanced for better accuracy.
    
    Parameters:
    -----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    y_train : np.ndarray or pd.Series
        Training labels
    
    Returns:
    --------
    SVC : Best model found
    """
    print("\n  Tuning SVM hyperparameters...")
    
    param_dist = {
        'C': [0.1, 0.5, 1, 5, 10, 50, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'class_weight': ['balanced', None]
    }
    
    svm = SVC(random_state=42, probability=True)
    
    random_search = RandomizedSearchCV(
        svm,
        param_dist,
        n_iter=20,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"  Best params: {random_search.best_params_}")
    print(f"  Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


def tune_xgboost(X_train, y_train):
    """
    Tune XGBoost hyperparameters using RandomizedSearchCV.
    Optimized for efficiency and accuracy.
    
    Parameters:
    -----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    y_train : np.ndarray or pd.Series
        Training labels
    
    Returns:
    --------
    XGBClassifier : Best model found
    """
    if not XGBOOST_AVAILABLE:
        print("\n  XGBoost not available. Skipping tuning.")
        return None
    
    print("\n  Tuning XGBoost hyperparameters...")
    
    param_dist = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.15],
        'n_estimators': [100, 150],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9]
    }
    
    xgb = XGBClassifier(
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )
    
    random_search = RandomizedSearchCV(
        xgb,
        param_dist,
        n_iter=12,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"  Best params: {random_search.best_params_}")
    print(f"  Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_models(X_train, y_train, enable_tuning=True):
    """
    Train multiple classification models with optional hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    y_train : np.ndarray or pd.Series
        Training labels
    enable_tuning : bool
        Whether to perform hyperparameter tuning (default: True)
    
    Returns:
    --------
    dict : Dictionary containing trained models
    """
    print("\n" + "=" * 70)
    print("TRAINING MODELS" + (" WITH HYPERPARAMETER TUNING" if enable_tuning else ""))
    print("=" * 70)
    
    models = {}
    
    # Decision Tree Classifier
    print("\n[1/4] Training Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    models['Decision Tree'] = dt_model
    print("✓ Decision Tree trained")
    
    # Random Forest Classifier (with optional tuning)
    print("[2/4] Training Random Forest Classifier...")
    if enable_tuning:
        rf_model = tune_random_forest(X_train, y_train)
    else:
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    print("✓ Random Forest trained")
    
    # Support Vector Machine (with optional tuning)
    print("[3/4] Training Support Vector Machine (SVM)...")
    if enable_tuning:
        svm_model = tune_svm(X_train, y_train)
    else:
        svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42
        )
        svm_model.fit(X_train, y_train)
    models['SVM'] = svm_model
    print("✓ SVM trained")
    
    # XGBoost Classifier (if available)
    if XGBOOST_AVAILABLE:
        print("[4/4] Training XGBoost Classifier...")
        if enable_tuning:
            xgb_model = tune_xgboost(X_train, y_train)
        else:
            xgb_model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train, y_train)
        
        if xgb_model is not None:
            models['XGBoost'] = xgb_model
            print("✓ XGBoost trained")
    
    # Gradient Boosting Classifier
    print("[5/5] Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    models['Gradient Boosting'] = gb_model
    print("✓ Gradient Boosting trained")
    
    # Create Voting Classifier Ensemble (Combined Best Models)
    print("\n[ENSEMBLE] Creating Voting Classifier...")
    voting_estimators = [
        ('rf', models['Random Forest']),
        ('svm', models['SVM']),
        ('gb', models['Gradient Boosting'])
    ]
    
    if 'XGBoost' in models:
        voting_estimators.append(('xgb', models['XGBoost']))
    
    voting_clf = VotingClassifier(
        estimators=voting_estimators,
        voting='soft'  # Use probability predictions
    )
    voting_clf.fit(X_train, y_train)
    models['Voting Ensemble'] = voting_clf
    print("✓ Voting Ensemble trained")
    
    return models


# ============================================================================
# MODEL EVALUATION FUNCTIONS
# ============================================================================

def evaluate_models(models, X_test, y_test, fertilizer_encoder):
    """
    Evaluate all trained models and select the best one.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : np.ndarray or pd.DataFrame
        Test features
    y_test : np.ndarray or pd.Series
        Test labels
    fertilizer_encoder : LabelEncoder
        Fertilizer label encoder
    
    Returns:
    --------
    tuple : (best_model, best_model_name, results_dict)
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    results = {}
    best_accuracy = 0
    best_model = None
    best_model_name = None
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 50)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'model': model
        }
        
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred,
            target_names=fertilizer_encoder.classes_,
            zero_division=0
        ))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Update best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name
    
    return best_model, best_model_name, results


def print_feature_importance(best_model, best_model_name, feature_names):
    """
    Print feature importance for tree-based models (Decision Tree, Random Forest).
    
    Parameters:
    -----------
    best_model : sklearn model
        Trained model
    best_model_name : str
        Name of the model
    feature_names : list
        Names of features
    """
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    
    if hasattr(best_model, 'feature_importances_'):
        print(f"\nFeature Importance ({best_model_name}):")
        print("-" * 50)
        
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        for i in range(len(feature_names)):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    else:
        print(f"\n{best_model_name} does not have feature importance attribute.")


# ============================================================================
# MODEL SAVING FUNCTIONS
# ============================================================================

def save_models(best_model, crop_encoder, fertilizer_encoder, scaler, output_dir='.'):
    """
    Save trained model and encoders to disk using joblib.
    
    Parameters:
    -----------
    best_model : sklearn model
        Best trained model
    crop_encoder : LabelEncoder
        Crop label encoder
    fertilizer_encoder : LabelEncoder
        Fertilizer label encoder
    scaler : StandardScaler
        Feature scaler
    output_dir : str
        Directory to save models (default: current directory)
    """
    print("\n" + "=" * 70)
    print("SAVING MODELS AND ENCODERS")
    print("=" * 70)
    
    try:
        # Save main model
        model_path = os.path.join(output_dir, 'fertilizer_model.pkl')
        joblib.dump(best_model, model_path)
        print(f"✓ Model saved: {model_path}")
        
        # Save crop encoder
        crop_encoder_path = os.path.join(output_dir, 'crop_encoder.pkl')
        joblib.dump(crop_encoder, crop_encoder_path)
        print(f"✓ Crop encoder saved: {crop_encoder_path}")
        
        # Save fertilizer encoder
        fertilizer_encoder_path = os.path.join(output_dir, 'fertilizer_encoder.pkl')
        joblib.dump(fertilizer_encoder, fertilizer_encoder_path)
        print(f"✓ Fertilizer encoder saved: {fertilizer_encoder_path}")
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"✓ Scaler saved: {scaler_path}")
        
        print("\n✓ All models saved successfully!")
        
    except Exception as e:
        print(f"✗ Error saving models: {str(e)}")
        raise


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def load_prediction_models(model_dir='.'):
    """
    Load trained models and encoders for prediction.
    
    Parameters:
    -----------
    model_dir : str
        Directory containing saved models (default: current directory)
    
    Returns:
    --------
    tuple : (model, crop_encoder, fertilizer_encoder, scaler, selector)
    """
    model_path = os.path.join(model_dir, 'fertilizer_model.pkl')
    crop_encoder_path = os.path.join(model_dir, 'crop_encoder.pkl')
    fertilizer_encoder_path = os.path.join(model_dir, 'fertilizer_encoder.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    selector_path = os.path.join(model_dir, 'feature_selector.pkl')
    
    model = joblib.load(model_path)
    crop_encoder = joblib.load(crop_encoder_path)
    fertilizer_encoder = joblib.load(fertilizer_encoder_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path) if os.path.exists(selector_path) else None
    
    return model, crop_encoder, fertilizer_encoder, scaler, selector


def predict_fertilizer(N, P, K, pH, crop, model=None, crop_encoder=None,
                      fertilizer_encoder=None, scaler=None, selector=None, model_dir='.'):
    """
    Predict fertilizer recommendation based on soil parameters.
    
    Parameters:
    -----------
    N : float
        Nitrogen content
    P : float
        Phosphorus content
    K : float
        Potassium content
    pH : float
        Soil pH value
    crop : str
        Crop name
    model : sklearn model or None
        Trained model (if None, loads from disk)
    crop_encoder : LabelEncoder or None
        Crop encoder (if None, loads from disk)
    fertilizer_encoder : LabelEncoder or None
        Fertilizer encoder (if None, loads from disk)
    scaler : StandardScaler or None
        Feature scaler (if None, loads from disk)
    selector : SelectKBest or None
        Feature selector (if None, loads from disk)
    model_dir : str
        Directory containing saved models
    
    Returns:
    --------
    str : Predicted fertilizer recommendation
    """
    # Load models if not provided
    if model is None:
        model, crop_encoder, fertilizer_encoder, scaler, selector = load_prediction_models(model_dir)
    
    try:
        # Prepare input features - create a temporary dataframe for feature engineering
        temp_df = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'PH': [pH],
            'CROP': [crop]
        })
        
        # Apply advanced feature engineering
        temp_df = create_advanced_features(temp_df)
        
        # Handle NaN values
        temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in temp_df.select_dtypes(include=[np.number]).columns:
            temp_df[col].fillna(temp_df[col].median(), inplace=True)
        
        # Encode crop
        crop_encoded = crop_encoder.transform([crop])[0]
        
        # Create feature array with all engineered features
        feature_cols = ['N', 'P', 'K', 'PH', 'N_P_Ratio', 'N_K_Ratio', 'P_K_Ratio', 
                       'NPK_Sum', 'NPK_Balance', 'N_P_Product', 'N_K_Product', 
                       'P_K_Product', 'High_N', 'High_P', 'High_K', 'pH_Level', 'CROP']
        
        # Replace CROP with encoded value
        temp_df['CROP'] = crop_encoded
        
        # Extract features in correct order
        features = temp_df[feature_cols].values
        
        # Apply feature selection if selector exists
        if selector is not None:
            features = selector.transform(features)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Decode prediction
        fertilizer = fertilizer_encoder.inverse_transform([prediction])[0]
        
        return fertilizer
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def main():
    """
    Main pipeline function: Load, preprocess, train, evaluate, and save models.
    """
    print("\n" + "=" * 70)
    print("FERTILIZER RECOMMENDATION SYSTEM - ML PIPELINE")
    print("=" * 70)
    
    try:
        # ====== Step 1: Load Datasets ======
        datasets = load_datasets()
        
        # ====== Step 2: Merge Datasets ======
        merged_df = merge_datasets(datasets)
        
        # ====== Step 3: Normalize Numerical Features ======
        print("\n" + "=" * 70)
        print("FEATURE NORMALIZATION")
        print("=" * 70)
        merged_df = normalize_numerical_features(merged_df)
        print("✓ Numerical features normalized to 0-100 range")
        
        # ====== Step 3.5: Create Advanced Features ======
        print("\n" + "=" * 70)
        print("ADVANCED FEATURE ENGINEERING")
        print("=" * 70)
        merged_df = create_advanced_features(merged_df)
        print("✓ Advanced features created:")
        print("  - NPK Ratios (N_P, N_K, P_K)")
        print("  - NPK Balance Metrics (Sum, Balance Std Dev)")
        print("  - Nutrient Interactions (Products)")
        print("  - High Nutrient Flags (Binary)")
        print("  - pH Level Categorization")
        
        # ====== Step 5: Encode Categorical Features ======
        print("\n" + "=" * 70)
        print("FEATURE ENCODING")
        print("=" * 70)
        merged_df, crop_encoder, fertilizer_encoder = encode_categorical_features(merged_df)
        print("✓ Categorical features encoded")
        print(f"  - Unique crops: {len(crop_encoder.classes_)}")
        print(f"  - Unique fertilizers: {len(fertilizer_encoder.classes_)}")
        
        # ====== Step 6: Handle Missing Values ======
        print("\n" + "=" * 70)
        print("HANDLING MISSING VALUES")
        print("=" * 70)
        # Replace any NaN or Inf values with column median
        merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in merged_df.select_dtypes(include=[np.number]).columns:
            merged_df[col].fillna(merged_df[col].median(), inplace=True)
        print(f"✓ Missing values handled - NaN and Inf values replaced with column medians")
        
        # ====== Step 6: Separate Features and Target ======
        # Include engineered features for better model accuracy
        feature_cols = ['N', 'P', 'K', 'PH', 'N_P_Ratio', 'N_K_Ratio', 'P_K_Ratio', 
                       'NPK_Sum', 'NPK_Balance', 'N_P_Product', 'N_K_Product', 
                       'P_K_Product', 'High_N', 'High_P', 'High_K', 'pH_Level', 'CROP']
        X = merged_df[feature_cols].values
        y = merged_df['FERTILIZER'].values
        
        # ====== Step 7: Feature Selection ======
        print("\n" + "=" * 70)
        print("FEATURE SELECTION")
        print("=" * 70)
        # Select best 12 features using SelectKBest
        selector = SelectKBest(f_classif, k=min(12, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        print(f"✓ Selected {X_selected.shape[1]} best features using SelectKBest")
        print(f"  - Original features: {len(feature_cols)}")
        print(f"  - Selected features: {X_selected.shape[1]}")
        
        # ====== Step 8: Feature Scaling ======
        print("\n" + "=" * 70)
        print("FEATURE SCALING")
        print("=" * 70)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        print("✓ Features scaled using StandardScaler")
        print(f"  - Total features: {X_selected.shape[1]} (after feature selection)")
        
        # ====== Step 9: Train-Test Split ======
        print("\n" + "=" * 70)
        print("TRAIN-TEST SPLIT")
        print("=" * 70)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        print(f"✓ Train set size: {X_train.shape[0]} samples (80%)")
        print(f"✓ Test set size: {X_test.shape[0]} samples (20%)")
        
        # ====== Step 10: Train Models ======
        models = train_models(X_train, y_train, enable_tuning=True)
        
        # ====== Step 11: Evaluate Models ======
        best_model, best_model_name, results = evaluate_models(
            models, X_test, y_test, fertilizer_encoder
        )
        
        print("\n" + "=" * 70)
        print("BEST MODEL SELECTED")
        print("=" * 70)
        print(f"✓ Best Model: {best_model_name}")
        print(f"✓ Accuracy: {results[best_model_name]['accuracy']:.4f} " +
              f"({results[best_model_name]['accuracy']*100:.2f}%)")
        
        # ====== Step 12: Print Feature Importance ======
        selected_feature_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_cols[i] for i in selected_feature_indices]
        print_feature_importance(best_model, best_model_name, selected_feature_names)
        
        # ====== Step 13: Save Models (with selector) ======
        # Save selector along with other models
        selector_path = os.path.join('.', 'feature_selector.pkl')
        joblib.dump(selector, selector_path)
        print(f"✓ Feature selector saved: {selector_path}")
        save_models(best_model, crop_encoder, fertilizer_encoder, scaler)
        
        # ====== Step 14: Sample Predictions ======
        print("\n" + "=" * 70)
        print("SAMPLE PREDICTIONS")
        print("=" * 70)
        
        # Get available crops from encoder
        available_crops = crop_encoder.classes_[:5]  # Sample 5 crops
        
        sample_inputs = [
            {'N': 60, 'P': 20, 'K': 20, 'pH': 6.5, 'crop': available_crops[0]},
            {'N': 40, 'P': 40, 'K': 40, 'pH': 7.0, 'crop': available_crops[1] if len(available_crops) > 1 else available_crops[0]},
            {'N': 80, 'P': 40, 'K': 40, 'pH': 5.8, 'crop': available_crops[2] if len(available_crops) > 2 else available_crops[0]},
        ]
        
        for i, sample in enumerate(sample_inputs, 1):
            fertilizer = predict_fertilizer(
                sample['N'],
                sample['P'],
                sample['K'],
                sample['pH'],
                sample['crop'],
                model=best_model,
                crop_encoder=crop_encoder,
                fertilizer_encoder=fertilizer_encoder,
                scaler=scaler
            )
            
            print(f"\nSample {i}:")
            print(f"  Input  -> N={sample['N']}, P={sample['P']}, K={sample['K']}, " +
                  f"pH={sample['pH']}, Crop={sample['crop']}")
            print(f"  Output -> Recommended Fertilizer: {fertilizer}")
        
        # ====== Pipeline Complete ======
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nSaved artifacts:")
        print("  - fertilizer_model.pkl")
        print("  - crop_encoder.pkl")
        print("  - fertilizer_encoder.pkl")
        print("  - scaler.pkl")
        print("\nReady for Streamlit integration!")
        
    except Exception as e:
        print(f"\n✗ Pipeline Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
