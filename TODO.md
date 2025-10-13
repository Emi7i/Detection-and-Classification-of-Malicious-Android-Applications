# TODO

First, we need to combine both datasets into one dataset. 
Then I would like to separate them into 2 datasets, one for static analysis (which will be used by my binary model) and one for dynamic analysis (which will be used by the classification model).

## Prepare the dataset
- [x] Combine both datasets
- [x] Remove all rows and all columns with at least one missing value
- [x] Remove empty rows and columns from the dataset

Now we need to look at the values in the data to understand which collumns should stay and which should not. 
I plan to first do my best estimate of what is important and what isn't and later, after I get some results from the models, come back and refine this selection.

- [ ] Remove columns that we wont be using
- [ ] Separate the data by static and dynamic analysis 

- [ ] Make a base binary model to test the data

> NOTE: From here on out, all of the steps will be done to both of our datasets.

We will try to condense the data by looking for features which don't seem to apply to any or many of the apps. For example, if none of the apps use Permission X, then it is not going to have any predictive value and can be dropped from out dataset. Even if only a few apps have it, it probably won't be enough to be statistically significant.

- [ ] Remove columns that not many apps have the data for

> NOTE: This can actually hurt our model. We will be using this in the starting steps, and them when we have our models set up, come back and refine this selection.