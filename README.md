# tml_2025_task1_5
I implemented the RMIA attack. I trained two ResNet-18 models with different configurations on the non-member images from the Pub dataset.
I computed the RMIA score for each pub dataset sample, and based on these scores, the loss and the label for each sample, I trained a classifier. This classifier was then applied to the priv dataset to compute the membership confidence score.
