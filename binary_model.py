import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC as SVM
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_processing import PATH_TO_SAVE_STATIC

SHOW_GRAPS = False
RANDOM_STATE = 42
SHOW_DATA = False  

model_files = {
    'Logistic Regression': 'models/Fly/logistic_regression_tuned.pkl',
    'Random Forest': 'models/Fly/random_forest_tuned.pkl',
    'XGBoost': 'models/Fly/xgboost_tuned.pkl',
    'LightGBM': 'models/Fly/lightgbm_tuned.pkl',
    'SVM': 'models/Fly/support_vector_machine_tuned.pkl',
    'AdaBoost': 'models/Fly/adaboost_tuned.pkl',
    'Neural Network': 'models/Fly/neural_network_tuned.pkl'
}

def run_fly():
    df = pd.read_csv(PATH_TO_SAVE_STATIC)
    
    x = df.drop(['Malware'], axis=1)
    y = df['Malware']

    if SHOW_DATA:
        show_data(x, y)

    x = x.drop(['nr_permissions', 'normal', 'dangerous'], axis=1)

    # First lets start with Random Forest, XGBoost/LightGBM and Logical Regression

    print(df['Malware'].value_counts())
    # Here we get:
    #     Malware
    # 1    40850
    # 0    36480
    # Meaning we have 52.8% actual malware and 47.2%  benign apps. This is perfect as we almost have 50/50 balance.

    # Remove correlated features
    x = remove_correlated_features(x, threshold=0.8)

    # Split the data by 70%/15%/15% 
    x_train, validation_x, y_train, validation_y = train_test_split(x, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y, shuffle=True) # IMPORTANT: We need to shuffle the data

    # Split the remaining 30% into 15% for validation and 15% for testing
    x_val, x_test, y_val, y_test = train_test_split(validation_x, validation_y, test_size=0.5, random_state=RANDOM_STATE, stratify=validation_y)

    binary_cols = [c for c in x.columns if set(x[c].unique()) <= {0, 1}]
    numeric_cols = [c for c in x.columns if c not in binary_cols]

    # Scale only numeric columns
    scaler = StandardScaler()
    x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
    x_val[numeric_cols] = scaler.transform(x_val[numeric_cols])
    x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])

    # Feature selection - Remove useless columns
    x_train, x_val, x_test = find_useful_columns(x_train, x_val, x_test, binary_cols, numeric_cols)

    # Apply PCA
    # x_train, x_val, x_test = apply_pca(x, x_train, x_val, x_test)

    # Find the best parameters for each model
    # tune_all_models(x_train, y_train)
    
    # Train and evaluate models
    # model_results = train_and_evaluate_models(x_train, y_train, x_val, y_val)

    # Load and evaluate pre-tuned models
    model_results = load_and_test_models(x_test=x_test, y_test=y_test, SHOW_GRAPS=True)

    # Print summary comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"{'Model':<22} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    for name, acc, prec, rec, f1 in model_results:
        print(f"{name:<22} {acc:>10.2f}% {prec:>10.2f}% {rec:>10.2f}% {f1:>10.2f}%")
    

def load_and_test_models(x_test, y_test, SHOW_GRAPS=True):    
    # Select which models to load
    enabled_models = {
        'Logistic Regression': True,
        'Random Forest': True,
        'XGBoost': True,
        'LightGBM': True, 
        'SVM': True,
        'AdaBoost': True,
        'Neural Network': True
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
        y_pred = model.predict(x_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred) * 100
        recall = recall_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100

        print("\n--- RESULTS ---")
        print(f"Accuracy:  {acc:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"Recall:    {recall:.2f}%")
        print(f"F1-Score:  {f1:.2f}%")
        
        # Confusion matrix
        if SHOW_GRAPS:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malware'])
            disp.plot(cmap='Blues')
            plt.title(f'{name} - Confusion Matrix')
            plt.show()

        # Store results
        model_results.append([name, acc, precision, recall, f1])

    return model_results

def train_and_evaluate_models(x_train, y_train, x_val, y_val):
    # Define models
    models = [
        [LogisticRegression(random_state=RANDOM_STATE, max_iter=10000, C=0.1, penalty='l2', solver='lbfgs'), "Logistic Regression"], # Increasing the max_iter above 10000 does not improve the results much
        [RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE, max_depth=30, min_samples_split=2, min_samples_leaf=1), "Random Forest"],
        [xgb.XGBClassifier(objective="binary:logistic", random_state=RANDOM_STATE, colsample_bytree=0.3, learning_rate=0.3, max_depth=9, n_estimators=300), "XGBoost"],
        [lgb.LGBMClassifier(random_state=RANDOM_STATE, learning_rate=0.3, max_depth=-1, n_estimators=400, num_leaves=100, verbose=-1), "LightGBM"],
        [SVM(kernel='rbf', random_state=RANDOM_STATE, C=10, gamma=0.1, probability=False), "Support Vector Machine"],
        [AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE, learning_rate=1), "AdaBoost"],
        [MLPClassifier(hidden_layer_sizes=(200, 150, 75), max_iter=100, random_state=RANDOM_STATE, alpha=0.00085, activation='relu', early_stopping=True), "Neural Network"],
    ]

    model_results = []

    for model, name in models:
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print('='*50)
        print(model.get_params())
        
        # Train model
        model.fit(x_train, y_train)
        
        # Predict on validation set
        y_pred = model.predict(x_val)
        
        # Calculate metrics
        acc = accuracy_score(y_val, y_pred) * 100
        precision = precision_score(y_val, y_pred) * 100
        recall = recall_score(y_val, y_pred) * 100
        f1 = f1_score(y_val, y_pred) * 100
        
        print("\n--- RESULTS ---")
        print(f"Accuracy:  {acc:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"Recall:    {recall:.2f}%")
        print(f"F1-Score:  {f1:.2f}%")
        
        # Confusion matrix
        if SHOW_GRAPS:
            cm = confusion_matrix(y_val, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malware'])
            disp.plot(cmap='Blues')
            plt.title(f'{name} - Confusion Matrix')
            plt.show()

        # Store results
        model_results.append([name, acc, precision, recall, f1])

        # Save the model!!!
        model_filename = f"models/Fly/{name.replace(' ', '_').lower()}_tuned.pkl"
        os.makedirs('models/Fly', exist_ok=True)
        joblib.dump(model, model_filename)
        print(f"Model saved to: {model_filename}")

    return model_results

def find_useful_columns(x_train, x_val, x_test, binary_cols, numeric_cols):
    NUM_APPS = x_train.shape[0]
    MARGIN_OF_ERROR = 100 / (NUM_APPS ** 0.5)
    FEATURE_THRESHOLD = int((MARGIN_OF_ERROR / 100) * NUM_APPS)

    print(f"Training set size: {NUM_APPS} apps")
    print(f"Feature threshold: {FEATURE_THRESHOLD} apps ({MARGIN_OF_ERROR:.2f}% margin of error)")

    cols_to_drop = []

    # Check binary cols
    for col in binary_cols:
        count_ones = x_train[col].sum()
        if count_ones < FEATURE_THRESHOLD or count_ones > (NUM_APPS - FEATURE_THRESHOLD):
            cols_to_drop.append(col)

    # Check numeric features - drop if almost no variance
    for col in numeric_cols:
        if x_train[col].std() < 0.1:  # NOTE: Increasing this value did not prove to affect the results
            cols_to_drop.append(col)

    print(f"Binary cols dropped: {len([c for c in cols_to_drop if c in binary_cols])}")
    print(f"Numeric cols dropped: {len([c for c in cols_to_drop if c in numeric_cols])}")
    print(f"Total num of features dropped: {len(cols_to_drop)}")

    # Drop from all sets
    x_train = x_train.drop(columns=cols_to_drop)
    x_val = x_val.drop(columns=cols_to_drop)
    x_test = x_test.drop(columns=cols_to_drop)

    print(f"New shape: {x_train.shape}")
    return x_train, x_val, x_test

def apply_pca(x, x_train, x_val, x_test):
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    x_train = pca.fit_transform(x_train)
    x_val = pca.transform(x_val)
    x_test = pca.transform(x_test)

    print(f"Original features: {x.shape[1]}")
    print(f"After PCA: {x_train.shape[1]}")
    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    return x_train, x_val, x_test


def show_data(x, y, target_name='Malware'):
    # Combine x and y for correlation calculation
    data_for_viz = x.copy()
    data_for_viz[target_name] = y

    # Calculate correlation with target
    correlations = data_for_viz.corr()[target_name].abs().sort_values(ascending=False)

    print(f"Correlation with {target_name}:")
    print(correlations.drop(target_name))

    # Correlation heatmap
    features = correlations.drop(target_name).index.tolist()
    features.append(target_name)

    corr_matrix = data_for_viz[features].corr()

    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Here we see some corelation 
    # total_perm seems to be identical to nr_permissions
    # We could probably drop ACCES_FINE_LOCATION 
    # And we could probably just use dangerous and drop nr_permissions

def tune_all_models(x_train, y_train, cv=5, n_jobs=-1, verbose=2): 
    # NOTE: This function works only because we do shuffle=True in train_test_split. Using it with the whole datasets REQUIRES shuffling!

    # Define models and their parameters that we are trying to tune
    model_params = {
        "Logistic Regression": {
            'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=10000),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'saga']
            }
        },
        
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
            'model': SVM(random_state=RANDOM_STATE),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
        },
        
        "AdaBoost": {
            'model': AdaBoostClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0]
            }
        },
        
       "Neural Network": {
            'model': MLPClassifier(random_state=RANDOM_STATE, max_iter=1000, early_stopping=True),
            'params': {
                'hidden_layer_sizes': [
                    (100,), (150,), (200,),                    # Single layer
                    (150, 75), (200, 100), (256, 128),         # Two layers
                    (150, 100, 50), (200, 150, 75)             # Three layers
                ],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
            }
        }
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
            scoring='f1',
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        grid_search.fit(x_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        best_models[name] = grid_search.best_estimator_
    
    for model in best_models.values():
        print(model)


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