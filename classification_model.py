import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC as SVM
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier

from binary_model import RANDOM_STATE, show_data, apply_pca
from data_processing import PATH_TO_SAVE_DYNAMIC

SHOW_DATA = False   
SHOW_GRAPS = False

model_files = {
    'Logistic Regression': 'models/Dragon/logistic_regression_tuned.pkl',
    'Random Forest': 'models/Dragon/random_forest_tuned.pkl',
    'XGBoost': 'models/Dragon/xgboost_tuned.pkl',
    'LightGBM': 'models/Dragon/lightgbm_tuned.pkl',
    'SVM': 'models/Dragon/support_vector_machine_tuned.pkl',
    'AdaBoost': 'models/Dragon/adaboost_tuned.pkl',
    'TabNet': 'models/Dragon/tabnet_tuned.pkl'
}

def run_dragon():
    df = pd.read_csv(PATH_TO_SAVE_DYNAMIC).drop(['nr_permissions', 'normal', 'dangerous'], axis=1)
    
    # Filter df to keep only malware samples
    df = df[df['Malware'] == 1]
    # Group families with less than 100 samples into 'Other'
    family_counts = df['MalFamily'].value_counts()
    df['MalFamily'] = df['MalFamily'].where(df['MalFamily'].map(family_counts) >= 100, 'Other')
    print(f"Number of classes after grouping: {df['MalFamily'].nunique()}")

    le = LabelEncoder()
    x = df.drop(['MalFamily', 'Malware'], axis=1)
    y = df['MalFamily']
    y_encoded = le.fit_transform(y)

    x = remove_correlated_features(x, threshold=0.95)

    binary_cols = [c for c in x.columns if set(x[c].unique()) <= {0, 1}]
    numeric_cols = [c for c in x.columns if c not in binary_cols]

    # Feature selection - Remove useless columns
    x = find_useful_columns(x, binary_cols, numeric_cols)

    if SHOW_DATA:
        show_data(x, y_encoded)
        print(df['MalFamily'].value_counts())
        # Here we get:
        #     MalFamily
        # 1    40850
        # 0    36480
        # Meaning we have 52.8% actual malware and 47.2%  benign apps. This is perfect as we almost have 50/50 balance.

    # Split the data by 70%/15%/15% 
    x_train, validation_x, y_train, validation_y = train_test_split(x, y_encoded, test_size=0.3, random_state=RANDOM_STATE, stratify=y_encoded, shuffle=True) # IMPORTANT: We need to shuffle the data

    # Split the remaining 30% into 15% for validation and 15% for testing
    x_val, x_test, y_val, y_test = train_test_split(validation_x, validation_y, test_size=0.5, random_state=RANDOM_STATE, stratify=validation_y, shuffle=True)

    binary_cols = [c for c in x.columns if set(x[c].unique()) <= {0, 1}]
    numeric_cols = [c for c in x.columns if c not in binary_cols]

    # Scale only numeric columns
    scaler = StandardScaler()
    x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
    x_val[numeric_cols] = scaler.transform(x_val[numeric_cols])
    x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])

    # Apply PCA
    # x_train, x_val, x_test = apply_pca(x, x_train, x_val, x_test)

    # Train and evaluate models
    # model_results = train_and_evaluate_models(x_train, y_train, x_val, y_val, labels=le.classes_)

    # Tune models
    # tune_all_models(x_train, y_train)

    # Load and evaluate pre-tuned models
    model_results = load_and_test_models(x_val, y_val, labels=le.classes_, SHOW_GRAPS=False)

    # Print summary comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"{'Model':<22} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    for name, acc, prec, rec, f1 in model_results:
        print(f"{name:<22} {acc:>10.2f}% {prec:>10.2f}% {rec:>10.2f}% {f1:>10.2f}%")


def find_useful_columns(x, binary_cols, numeric_cols):
    NUM_APPS = x.shape[0]
    MARGIN_OF_ERROR = 100 / (NUM_APPS ** 0.5)
    FEATURE_THRESHOLD = int((MARGIN_OF_ERROR / 100) * NUM_APPS)

    print(f"Training set size: {NUM_APPS} apps")
    print(f"Feature threshold: {FEATURE_THRESHOLD} apps ({MARGIN_OF_ERROR:.2f}% margin of error)")

    cols_to_drop = []

    # Check binary cols
    for col in binary_cols:
        count_ones = x[col].sum()
        if count_ones < FEATURE_THRESHOLD or count_ones > (NUM_APPS - FEATURE_THRESHOLD):
            cols_to_drop.append(col)

    # Check numeric features - drop if almost no variance
    for col in numeric_cols:
        if x[col].std() < 0.3:  # NOTE: Increasing this value did not prove to affect the results
            cols_to_drop.append(col)

    print(f"Binary cols dropped: {len([c for c in cols_to_drop if c in binary_cols])}")
    print(f"Numeric cols dropped: {len([c for c in cols_to_drop if c in numeric_cols])}")
    print(f"Total num of features dropped: {len(cols_to_drop)}")

    # Drop from all sets
    x = x.drop(columns=cols_to_drop)

    print(f"New shape: {x.shape}")
    return x
    

def load_and_test_models(x_val, y_val, labels, SHOW_GRAPS=True):    
    # Select which models to load
    enabled_models = {
        'Logistic Regression': True,
        'Random Forest': True,
        'XGBoost': True,
        'LightGBM': True, 
        'SVM': True,
        'AdaBoost': True,
        'TabNet': True,
    }

    # Load models
    models = [
        [joblib.load(model_files[name]), name] 
        for name, enabled in enabled_models.items() 
        if enabled
    ]

    model_results = []

    for model, name in models:
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print('='*50)
        
        # Predict on validation set
        y_pred = model.predict(x_val.values if name == "TabNet" else x_val)
        
        # Calculate metrics
        acc = accuracy_score(y_val, y_pred) * 100
        precision = precision_score(y_val, y_pred, average='weighted') * 100
        recall = recall_score(y_val, y_pred, average='weighted') * 100 
        f1 = f1_score(y_val, y_pred, average='weighted') * 100  

        print("\n--- RESULTS ---")
        print(f"Accuracy:  {acc:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"Recall:    {recall:.2f}%")
        print(f"F1-Score:  {f1:.2f}%")
        
        #Confusion matrix
        if SHOW_GRAPS:
            cm = confusion_matrix(y_val, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            fig, ax = plt.subplots(figsize=(12, 10))
            disp.plot(cmap='Blues', ax=ax,  xticks_rotation='vertical')
            plt.title(f'{name} - Confusion Matrix')
            plt.show()

        # Store results
        model_results.append([name, acc, precision, recall, f1])

    return model_results


def train_and_evaluate_models(x_train, y_train, x_val, y_val, labels):
    # Define models
    models = [
        [LogisticRegression(random_state=RANDOM_STATE, max_iter=10000, C=52, penalty='l2', multi_class='ovr'), "Logistic Regression"],
        [RandomForestClassifier(max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=120), "Random Forest"], 
        [xgb.XGBClassifier(objective="multi:softmax", random_state=RANDOM_STATE), "XGBoost"], 
        [lgb.LGBMClassifier(random_state=RANDOM_STATE, objective='multiclass', num_class=36, class_weight='balanced', num_leaves=64, max_depth=9, n_estimators=400, learning_rate=0.095,  verbose=-1), "LightGBM"],
        [SVM(kernel='rbf', random_state=RANDOM_STATE, decision_function_shape='ovr', C=250, gamma='scale', probability=False), "Support Vector Machine"], 
        [AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE), n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE), "AdaBoost"],
        [TabNetClassifier(verbose=0, seed=RANDOM_STATE, n_d=32, n_a=32, n_steps=3), "TabNet"],
    ]

    model_results = []

    for model, name in models:
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print('='*50)
        print(model.get_params())
        
        # Train model
        if name == "TabNet":
            model.fit(
                X_train=np.array(x_train),
                y_train=y_train,
                eval_set=[(np.array(x_val), y_val)],
                eval_name=['val'],
                eval_metric=['accuracy'],
                max_epochs=100,
                patience=10,
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )
        else:
            model.fit(x_train, y_train)
    
        # Predict
        y_pred = model.predict(np.array(x_val))
        
        # Calculate metrics
        acc = accuracy_score(y_val, y_pred) * 100
        precision = precision_score(y_val, y_pred, average='weighted') * 100
        recall = recall_score(y_val, y_pred, average='weighted') * 100 
        f1 = f1_score(y_val, y_pred, average='weighted') * 100  
        
        print("\n--- RESULTS ---")
        print(f"Accuracy:  {acc:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"Recall:    {recall:.2f}%")
        print(f"F1-Score:  {f1:.2f}%")
        
        # Confusion matrix
        if SHOW_GRAPS:
            cm = confusion_matrix(y_val, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            fig, ax = plt.subplots(figsize=(12, 10))
            disp.plot(cmap='Blues', ax=ax,  xticks_rotation='vertical')
            plt.title(f'{name} - Confusion Matrix')
            plt.show()

        # Store results
        model_results.append([name, acc, precision, recall, f1])

        # Save the model!!!
        model_filename = f"models/Dragon/{name.replace(' ', '_').lower()}_tuned.pkl"
        os.makedirs('models/Dragon', exist_ok=True)
        joblib.dump(model, model_filename)
        print(f"Model saved to: {model_filename}")

    return model_results


def remove_correlated_features(x, threshold=0.95):
    # Calculate correlation matrix on training data
    corr_matrix = x.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"\nRemoving {len(to_drop)} highly correlated features (threshold={threshold}):")
    print(to_drop)
    
    # Drop from all sets
    x = x.drop(columns=to_drop)
    
    print(f"Remaining features: {x.shape[1]}")
    
    return x


def tune_all_models(x_train, y_train, cv=5, n_jobs=-1, verbose=2): 
    # NOTE: This function works only because we do shuffle=True in train_test_split. Using it with the whole datasets REQUIRES shuffling!

    # Define models and their parameters that we are trying to tune
    model_params = {  
        "Random Forest": {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        
        "XGBoost": {
            'model': xgb.XGBClassifier(objective="binary:logistic", random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        },
        
        "LightGBM": {
            'model': lgb.LGBMClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, -1],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        "Support Vector Machine": {
            'model': SVM(random_state=RANDOM_STATE, decision_function_shape='ovr', probability=False, kernel='rbf'),
            'params': {
                'C': [100, 150, 200, 250],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            }
        },
    }
    
    best_models = {}
    
    for name, mp in model_params.items():
        print(f"\n{'='*70}")
        print(f"Tuning: {name}")
        print('='*70)
        
        grid_search = GridSearchCV(
            estimator=mp['model'],
            param_grid=mp['params'],
            cv=cv,
            scoring='f1_weighted',
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        grid_search.fit(x_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        best_models[name] = grid_search.best_estimator_
    
    for model in best_models.values():
        print(model)