# TODO

First, we need to combine both datasets into one dataset. 
Then I would like to separate them into 2 datasets, one for static analysis (which will be used by my binary model) and one for dynamic analysis (which will be used by the classification model).

## Prepare the dataset
- [x] Combine both datasets
- [x] Remove all rows and all columns with at least one missing value
- [x] Remove empty rows and columns from the dataset

Now we need to look at the values in the data to understand which columns should stay and which should not. 
I plan to manually select the columns, but plan to come back and refine this selection in the future.

- [x] Separate the data by static and dynamic analysis
- [x] Remove columns that we wont be using
- [x] Remove colums that have all 0s or 1s

## Models
> Lets name the models so we can refer to them more easily
 
I will call the binary (static) model `Fly`, and the classification (dynamic) model, `Dragon`. Together they make a `Dragonfly`. 

- For the `Dragon` model, we will first mesure its performance on just on the dynamic dataset, and then add the static dataset to see if the perfomance has changed significantly.
- For the `Fly` model, I will try to first use just the count of the "dangerous" and "normal" permissions, so we can see how close can it predict the apps on just those values alone (As they could be easily extracted). 

## Fly model (binary, static)
- [x] Load data
- [x] Use PCA to refine data
- [x] Train and test some models

Start model parameters: 
```python
        [LogisticRegression(random_state=42, max_iter=10000), "Logistic Regression"],
        [RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"],
        [xgb.XGBClassifier(objective="binary:logistic", random_state=42), "XGBoost"],
        [lgb.LGBMClassifier(random_state=42), "LightGBM"]
        [SVM(kernel='rbf', random_state=42), "Support Vector Machine"],
        [AdaBoostClassifier(n_estimators=100, random_state=42), "AdaBoost"],
        [MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42), "Neural Network"],
```

### First results: 
Okay so I got:
```c
    ======================================================================
    Model                  Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    Logistic Regression       93.26%      94.91%      92.18%      93.53%
    Random Forest             97.32%      97.58%      97.34%      97.46%
    XGBoost                   97.59%      97.76%      97.68%      97.72%
    LightGBM                  96.77%      97.05%      96.82%      96.94%
    Support Vector Machine    95.01%      96.89%      93.55%      95.19%
    AdaBoost                  93.19%      93.88%      93.18%      93.53%
    Neural Network            97.38%      97.35%      97.70%      97.52%
```

Meaning that our models are pretty good!
However, I realized that some of our data does not correlate to the `Malware` field at all!
```c
    Correlation with Malware:
        READ_PHONE_STATE                            0.659059
        dangerous                                   0.560881
        nr_permissions                              0.537815
        ACCESS_WIFI_STATE                           0.525105
        normal                                      0.492861
                                                    ...   
        REQUEST_COMPANION_USE_DATA_IN_BACKGROUND         NaN
        SMS_FINANCIAL_TRANSACTIONS                       NaN
        START_VIEW_PERMISSION_USAGE                      NaN
        WRITE_VOICEMAIL                                  NaN
        NrContactedIps                                   NaN
```
I need to go back and remove all of the rows and cols which have all 0s or 1s. 

- [x] Test models

### Testing
We could try to condense the data by looking for features which don't seem to apply to any or many of the apps. For example, if none of the apps use Permission X, then it is not going to have any predictive value and can be dropped from out dataset. Even if only a few apps have it, it probably won't be enough to be statistically significant.

- [x] Remove columns that not many apps have the data for 

> NOTE: This can actually hurt our model so we need to be careful

#### Results: 
```c
    ======================================================================
    RESULTS with PCA, and no selection of features
    ======================================================================
    Model                  Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    Logistic Regression       91.48%      92.68%      91.07%      91.87%
    Random Forest             97.12%      97.34%      97.21%      97.27%
    XGBoost                   97.22%      97.22%      97.52%      97.37%
    LightGBM                  96.69%      96.97%      96.75%      96.86%
    Support Vector Machine    94.26%      96.10%      92.90%      94.47%
    AdaBoost                  91.84%      92.92%      91.51%      92.21%
    Neural Network            96.70%      96.44%      97.34%      96.89%
```
```c
    ======================================================================
    RESULTS with feature selection, no PCA
    ======================================================================
    Model                  Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    Logistic Regression       93.92%      95.53%      92.83%      94.16%
    Random Forest             98.13%      98.21%      98.25%      98.23%
    XGBoost                   97.79%      97.97%      97.85%      97.91%
    LightGBM                  97.63%      97.92%      97.58%      97.75%
    Support Vector Machine    95.88%      97.00%      95.14%      96.06%
    AdaBoost                  94.09%      95.24%      93.47%      94.35%
    Neural Network            97.80%      98.13%      97.70%      97.91%
```
```c
    ======================================================================
    RESULTS with both feature selection and PCA
    ======================================================================
    Model                  Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    Logistic Regression       91.46%      92.64%      91.06%      91.84%
    Random Forest             97.07%      97.21%      97.24%      97.23%
    XGBoost                   97.38%      97.66%      97.37%      97.52%
    LightGBM                  96.53%      96.78%      96.65%      96.72%
    Support Vector Machine    94.22%      95.97%      92.97%      94.45%
    AdaBoost                  91.71%      92.95%      91.22%      92.08%
    Neural Network            96.95%      96.52%      97.75%      97.13%
```
We need scaling for most of our models, so lets see how scaling affects our best models which dont need it:
```c
    ======================================================================
    RESULTS with feature selection, but no scaling nor PCA 
    ======================================================================
    Model                    Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    XGBoost                   97.79%      97.97%      97.85%      97.91%
    LightGBM                  97.59%      97.81%      97.62%      97.71%
```
> NOTE: I will now focus on using just feature selection with scaling, without PCA. 

- [x] Tune the models using a function
```c
    ======================================================================
    RESULTS AFTER TUNING
    ======================================================================
    Model                    Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    Logistic Regression       93.79%      95.49%      92.62%      94.03%
    Random Forest             98.15%      98.32%      98.17%      98.24%
    XGBoost                   98.34%      98.26%      98.61%      98.44%
    LightGBM                  98.53%      98.74%      98.47%      98.60%
    Support Vector Machine    97.45%      97.65%      97.52%      97.58%
    AdaBoost                  94.64%      95.49%      94.30%      94.89%
    Neural Network            97.89%      98.44%      97.55%      97.99%
```
Looking at the F1-score, the most promising models are `LightGBM`, `XGBoost`, `Random Forest` and `MLP`

The problem I noticed with SVM, is that even if I try predicting the values, it takes a long time.

I will now try to manually go throught the models and fine-tune them to see if we can get to 99% F1-score
- [x] Fine tune the models manually

### Models after fine-tuning:
```python
    [RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE, max_depth=30, min_samples_split=2, min_samples_leaf=1), "Random Forest"],
    [xgb.XGBClassifier(objective="binary:logistic", random_state=RANDOM_STATE, colsample_bytree=0.3, learning_rate=0.3, max_depth=9, n_estimators=300), "XGBoost"],
    [lgb.LGBMClassifier(random_state=RANDOM_STATE, learning_rate=0.3, max_depth=-1, n_estimators=400, num_leaves=100), "LightGBM"],
    [MLPClassifier(hidden_layer_sizes=(200, 150, 75), max_iter=100, random_state=RANDOM_STATE, alpha=0.00085, activation='relu', early_stopping=True), "Neural Network"]
```

#### Results: 
```c
    ======================================================================
    MODEL COMPARISON
    ======================================================================
    Model                  Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    Random Forest             98.15%      98.32%      98.17%      98.24%
    XGBoost                   98.50%      98.58%      98.58%      98.58%
    LightGBM                  98.61%      98.77%      98.60%      98.68%
    Neural Network            97.98%      98.25%      97.93%      98.09%
```
However, in practice, If we use `total_perm` instead of `dangerous` and `normal`, we get: 
```c
    ======================================================================
    MODEL COMPARISON
    ======================================================================
    Model                  Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    Logistic Regression       93.78%      95.44%      92.64%      94.02%
    Random Forest             98.16%      98.33%      98.17%      98.25%
    XGBoost                   98.39%      98.62%      98.32%      98.47%
    LightGBM                  98.55%      98.74%      98.51%      98.63%
    Support Vector Machine    97.43%      97.65%      97.49%      97.57%
    AdaBoost                  94.53%      95.18%      94.42%      94.80%
    Neural Network            98.02%      98.06%      98.19%      98.12%
```
> But still, LightGBM wins!

### Testing the final model:
```c
    ======================================================================
    Model                  Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    LightGBM                  98.37%      98.82%      98.09%      98.45%
```
[Confusion Matrix](Binary_res.png)


## Dragon model (classification, dynamic)

- [ ] Load data
