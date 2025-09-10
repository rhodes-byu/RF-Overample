# RF-Overample

The project aims to address the class imbalance problem through synthetic point generation. One popular method for this currently is SMOTE which involves generating similar points by taking random "probable" values bewteen two points to synthetically generate another. However, while commonly accepted, and seems to help in rebalancing the dataset by augmenting the minority class with synthetic points. These points are nothing more than interpolations between existing points. Which is to say, they may lack understanding when it comes to encoded categorical features (ex. count encoded, it may mistakenly think green (4) is halfway between blue (2) and red (6). This can problem can be mitigated by using onehot encoding and taking the argmax to find the proper value for synthetically generated points such that they aren't invalid points in the categorical space. (.25 blue, .25 red, .5 green, isn't a valid onehot encoded variable. It should be (0 blue, 0 red, 1 green). Therefore in order to address these issues, we aim to propose and explore the effectiveness of random forest proximities in point generation.

Random forest proximities are essentially values that quantify how similar one point is to another. These proximities are calculated by determining through a random forest how often each point ends up in the same tree as other points as a metric to determine similarity. Since this metric doesn't involve interpolation, and is already using what a random forest is already doing, we aim to use random forest proximities in point generation to generate smarter points that can better follow the natural shape of the data. This is done using the RFgap module. The RFoversample then uses this RFgap as a policy for synthetic point generation.

Our primary metric for performance is weighted F1-score. This is calculated by dividing the percentage of the scores among the classes irregardless of points within each class. In other words, if a dataset has 4 classes, 25% of the score will be fore class 1, 25% for class 2, ect... Such that if the model is predicting well only on the majority class, but not other classes. It will only get 25% weighted F1-score, even if it technically gets 90% of the test set correct. 

Our primary means of testing is the comparison of methods "Baseline, None, RFoversample, SMOTE, Class Weights".

They were  initially compared across metrics of Dataset, Imbalance Ratio, Seed Counts, Encoding Methods, and Archetype Use, Archetype Proportion, and Reintroduced Minority

Dataset refered to the Dataset being tested. These are in the datasets folder, and consist of various datasets from the UCI Machine Learning Repository. They have been formatted such that the target is the first column.

Imbalance Ratio refers to the imbalance between the Majority and Minority classes. This logic can be found in Imbalancer.py with the SupportFunctions folder. The imbalance proportion refers to the percent of the data that will belong to each minority classes with respect to the majority class. In other words, if the training set consists of 300 class A, 200 class B, 16 class C. If our imbalancer proportion is 0.1, for 10% of the data will be minority class. Then we will retain our 300 training samples from the majority class, and then choose up to 30 samples for class B, and up to 30 samples for class C. However since class C has fewer than 30 total samples, it will instead choose all 16 that it does have. This is to ensure stability in method resampling approaches. (particularly those involving K nearest neighbors). The points are sampled randomly from their class.

Seed Counts refers to the global seed. In mixedrunner.py, we set a number of global random seeds. The experiment will iterate through each random seed. This will allow us to use the mean and standard error of our experiment results to quantify consistent performance for one method over another. The seeds are set such that each method should be working with same resampled data. This is also to make the experiment reproducible.

Encoding Methods refers to which method for encoding categorical variables was used. This refers to Ordinal Encoding, or Onehot encoding. Although later Rfoversample will internally do onehot encoding such that it can properly remember column names. It accepts categorical column headings as input for those that need to be cateogrical columns. These are identified by non-numeric data type values of columns.

Archetype Use, Archetype Proportion, and Reintroduced Minority were experimented with in runner.py, however, they were largely found ineffective and have since been depreciataed while still present in runner.py. Archetype use refers to whether or not to use archetypes (boolean), if true, Archetype proportion refers to the percent of the training sets minority class to create archetypal points. In other words, if the minority class has 30 points, and archetype proportion is 0.1, there will be 3 archetypes from those points. Reintroduced Minority, refers to the percentage of minority class points we will reintroduce along with our archetypes before passing it into our random forest. This is to ensure stability among KNN approaches. 

We also experimented with a proximity outlier based approach, using the random forest proximities to find points not similar to other points to help improve variability, however, it often led to the model overgeneralizing and distorting the original distribution of the features in the minority class, one poor outlier potentially skewing the synthetic point generation negatively.

We found that the random forest performed consistently better as we reintroduced more minority points back into the equation. There was no significant improvement when archetypes were generated and then added to minority class (no further reduction of points). Thus we concluded that archetypes were not helpful to our current implementation.

The past workflow (involving archetypes) primarily used runner.py and primarily cross_val.py among the other files in the SupportFunctions folder.
In our current workflow implementation, I often use mixedrunner.py along with mixedresults.py and mixedresults2.py.

Typically, load_datasets.py will function to create a pickle file of datasets called prepared_datasets.pkl. This pickle will then be fed into mixedrunner.py. This was initially done so that we could manually identify features that should be treated categorically for each dataset. 

As we added datasets from the OpenML-18 database, it was necessary to create OPENMLDatasetFetcher.py which grabs the datasets using their API (to rerun you will need to add your own API key) and ml18cleaner, which served to identify datasets with missing values, and examine how onehot encoding would effect each datasets. (remove or revise datasets where onehot encoding leads to explosive dimensionality).

Within the datasets, there are the generic UCI datasets, openml-cc18 and inactiveopenml-cc18 datasets. The ones removed were removed due to either explosive dimensionality or missing values although they could be added back in with a more robust preparation process.

We are currently experimenting with a revised rf_oversampler in the mixedrunner.py relying on the rfoversamplerJ framework.

This file uses K nearest neighbors to identify a number of proximity neighbors to consider when generating a point. (As you add more neighbors, the generated points will be more similar to one another, while lower will cause them to be less similar. Lower can potentially capture the unique shape of the data better, but higher is less prone to outliers).

It also uses gaussion noise generated from the covariance matrix to help shift the generated points slightly and prevent them from becoming to similar to one another. The strenght of this nosie can be controled with the noise scale. Noise and strength can be adjusted. (True/False) (default 0.15)

We are also experimenting with the option of categorical probability sampling. If enabled, when points are generated there will be probabilites of belonging to each onehot encoded variable. If false, we will use the argmax function to take the one with the highest probability. If true, we will sample from the probabilites, potentially allowing the data to be more diverse in cases where two potential options are close in probability.

There is also an enforce domains boolean. When true, this will make it so synthetically generated points will retain feature values within the bounds of their neighbors. Which is to say, if a group of points (based upon neighbors) has a max age of 30 and min age of 15. The generated point, if outside these bounds due to noise, will be rounded back into them. ex. 32 -> 30.

There is also hybrid_perturb_frac, which refers to the precentage of points that instead of using a neighbors based approach, will simply be duplicated from an existing point, and then adding random noise. This is meant to stop the points from generalizing too easily.

There is also boundary strategy. Which refers to how new points will be considered. If boundary = majority, each non majority class will generate points along the neighbors boundary between their respective class and the majority. If boundary = nearest, each non majority class will instead look for the nearest opposing class and generate points along that boundary. This is meant to help in cases where 2 class may be close in proximities, but neither is the majority class. In theory points will then be generated in more effective regions. 

<img width="1600" height="1260" alt="summary_delta_vs_smote" src="https://github.com/user-attachments/assets/505508b5-4bba-4ab2-8f99-4da1caa21b3f" />

This is an example graph using the averaged weighted F1 score over 15 seeds on many of the original datasets showing the potential for RFoversample as an alternative for SMOTE, and the potential for neighbors based approaches in the class imbalance space.
