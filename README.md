# tml_2025_task1_5
My idea was to implement an MIA using the features output by the ResNet-18 model for new datasets, and to train a classifier that uses these features to infer membership.
For the features, I selected confidence, as mentioned in the lecture. Loss is used by many different MIAs, as well as entropy. 
Finally, I used the 44 labels that the ResNet-18 model outputs at the end of each training. model.eval() (line 30) is used to put the model into evaluation mode, to ensure that the model behaves consistently and that the features can be extracted.

To find the best results, I tried two classifiers: Random Forest and XGBoost. I trained them with different parameters and found that XGBoost produced the best results.
Regarding the features, I experimented with various combinations of the four features and achieved optimal results with all four.

In the code, 'extract_features' extracts all the features from the 'pub' dataset in order to train the classifier. 
Extract_features is then used to extract all the features for the priv dataset.
The attack model, defined in line 96, uses these features to predict the probability (line 124) that each image in the priv dataset is a member.

In lines 92 and 93, and 122, any missing values in both datasets are replaced by the mean of the corresponding column. This ensures a complete dataset. 
Before each dataset is used, it is normalised to have the same characteristics as the training data.
