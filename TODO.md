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
- [ ] Remove colums that have all 0s or 1s

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
    Model              Accuracy     Precision    Recall       F1-Score
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

### Testing
We could try to condense the data by looking for features which don't seem to apply to any or many of the apps. For example, if none of the apps use Permission X, then it is not going to have any predictive value and can be dropped from out dataset. Even if only a few apps have it, it probably won't be enough to be statistically significant.

- [ ] Remove columns that not many apps have the data for 

> NOTE: This can actually hurt our model so we need to be careful

I plan to try this out with and without PCA to see how the models perform. 
