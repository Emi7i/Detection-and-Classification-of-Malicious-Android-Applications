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
```css
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
I need to go back and remove all of the rows and cols which have all 0s or 1s. 

- [x] Test models

### Testing
We could try to condense the data by looking for features which don't seem to apply to any or many of the apps. For example, if none of the apps use Permission X, then it is not going to have any predictive value and can be dropped from out dataset. Even if only a few apps have it, it probably won't be enough to be statistically significant.

- [x] Remove columns that not many apps have the data for 

> NOTE: This can actually hurt our model so we need to be careful

#### Results: 
```css
    ======================================================================
    TEST RESULTS with PCA, and no selection of features
    ======================================================================
    Model                  Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    Logistic Regression       91.41%      92.33%      91.33%      91.83%
    Random Forest             97.26%      97.67%      97.13%      97.40%
    XGBoost                   97.36%      97.66%      97.34%      97.50%
    LightGBM                  96.64%      96.97%      96.65%      96.81%
    Support Vector Machine    94.13%      95.90%      92.85%      94.35%
    AdaBoost                  91.10%      92.42%      90.58%      91.49%
    Neural Network            96.92%      96.68%      97.52%      97.10%
```
```css
    ======================================================================
    TEST RESULTS with feature selection, no PCA
    ======================================================================
    Model                  Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    Logistic Regression       93.94%      95.57%      92.84%      94.18%
    Random Forest             97.95%      98.34%      97.76%      98.05%
    XGBoost                   97.96%      98.44%      97.68%      98.06%
    LightGBM                  97.53%      97.89%      97.44%      97.66%
    Support Vector Machine    95.76%      96.87%      95.04%      95.95%
    AdaBoost                  94.11%      95.43%      93.33%      94.37%
    Neural Network            97.71%      98.00%      97.65%      97.83%
```
```css
    ======================================================================
    TEST RESULTS with both feature selection and PCA
    ======================================================================
    Model                  Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    Logistic Regression       91.42%      92.36%      91.32%      91.84%
    Random Forest             97.16%      97.52%      97.10%      97.31%
    XGBoost                   97.33%      97.67%      97.26%      97.47%
    LightGBM                  96.69%      97.07%      96.65%      96.86%
    Support Vector Machine    94.05%      95.77%      92.84%      94.28%
    AdaBoost                  90.97%      92.53%      90.19%      91.35%
    Neural Network            96.94%      96.71%      97.52%      97.12%
```
We need scaling for most of our models, so lets see how scaling affects our best models which dont need it:
```css
    ======================================================================
    TEST RESULTS with feature selection, but no scaling nor PCA 
    ======================================================================
    Model                    Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    XGBoost                   97.96%      98.44%      97.68%      98.06%
    LightGBM                  97.58%      97.95%      97.45%      97.70%
```
> NOTE: I will now focus on using just feature selection with scaling, without PCA. 

- [ ] Tune the models

