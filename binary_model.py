import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC as SVM
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_processing import PATH_TO_SAVE_STATIC

def run_fly():
    df = pd.read_csv(PATH_TO_SAVE_STATIC)
    
    x = df.drop(['Malware'], axis=1)
    y = df['Malware']

    #show_data(x, y)
    #Correlation with Malware:
    # READ_PHONE_STATE                            0.659059
    # dangerous                                   0.560881
    # nr_permissions                              0.537815
    # ACCESS_WIFI_STATE                           0.525105
    # normal                                      0.492861
    #                                             ...   
    # REQUEST_COMPANION_USE_DATA_IN_BACKGROUND         NaN
    # SMS_FINANCIAL_TRANSACTIONS                       NaN
    # START_VIEW_PERMISSION_USAGE                      NaN
    # WRITE_VOICEMAIL                                  NaN
    # NrContactedIps                                   NaN

    x = x.drop(['total_perm', 'nr_permissions'], axis=1)

    # First lets start with Random Forest, XGBoost/LightGBM and Logical Regression

    print(df['Malware'].value_counts())
    # Here we get:
    #     Malware
    # 1    40850
    # 0    36480
    # Meaning we have 52.8% actual malware and 47.2%  benign apps. This is perfect as we almost have 50/50 balance.

    # Split the data by 70%/15%/15% 
    x_train, validation_x, y_train, validation_y = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y, shuffle=True) # IMPORTANT: We need to shuffle the data

    # Split the remaining 30% into 15% for validation and 15% for testing
    x_val, x_test, y_val, y_test = train_test_split(validation_x, validation_y, test_size=0.5, random_state=42, stratify=validation_y)

    binary_cols = [c for c in x.columns if set(x[c].unique()) <= {0, 1}]
    numeric_cols = [c for c in x.columns if c not in binary_cols]

    # Scale only numeric columns
    scaler = StandardScaler()
    x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
    x_val[numeric_cols] = scaler.transform(x_val[numeric_cols])
    x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])

    # # Apply PCA
    # pca = PCA(n_components=0.98, random_state=42)
    # x_train = pca.fit_transform(x_train)
    # x_val = pca.transform(x_val)
    # x_test = pca.transform(x_test)

    # print(f"Original features: {x.shape[1]}")
    # print(f"After PCA: {x_train.shape[1]}")
    # print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # Define models
    models = [
        # [LogisticRegression(random_state=42, max_iter=10000), "Logistic Regression"],
        # [RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"],
        [xgb.XGBClassifier(objective="binary:logistic", random_state=42), "XGBoost"],
        [lgb.LGBMClassifier(random_state=42), "LightGBM"]
        
        # [SVM(kernel='rbf', random_state=42), "Support Vector Machine"],
        # [AdaBoostClassifier(n_estimators=100, random_state=42), "AdaBoost"],
        # [MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42), "Neural Network"],
    ]

    model_results = []

    for model, name in models:
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print('='*50)
        
        # Train model
        model.fit(x_train, y_train)
        
        # Predict on validation set
        y_pred = model.predict(x_val)
        
        # Calculate metrics
        acc = accuracy_score(y_val, y_pred) * 100
        precision = precision_score(y_val, y_pred) * 100
        recall = recall_score(y_val, y_pred) * 100
        f1 = f1_score(y_val, y_pred) * 100
        
        print(f"Accuracy:  {acc:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"Recall:    {recall:.2f}%")
        print(f"F1-Score:  {f1:.2f}%")
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malware'])
        disp.plot(cmap='Blues')
        plt.title(f'{name} - Confusion Matrix')
        plt.show()
        
        # Store results
        model_results.append([name, acc, precision, recall, f1])

    # Print summary comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"{'Model':<24} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    for name, acc, prec, rec, f1 in model_results:
        print(f"{name:<20} {acc:>10.2f}% {prec:>10.2f}% {rec:>10.2f}% {f1:>10.2f}%")


def show_data(x, y):
    # Combine x and y for correlation calculation
    data_for_viz = x.copy()
    data_for_viz['Malware'] = y

    # Calculate correlation with target
    correlations = data_for_viz.corr()['Malware'].abs().sort_values(ascending=False)

    print("Correlation with Malware:")
    print(correlations.drop('Malware'))

    # Correlation heatmap
    features = correlations.drop('Malware').index.tolist()
    features.append('Malware')

    corr_matrix = data_for_viz[features].corr()

    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
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