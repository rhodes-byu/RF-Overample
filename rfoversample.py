import numpy as np
import pandas as pd
from rfgap import RFGAP


class RFOversampler:

    """
    Currently (04-01-25) wrtten to upsample all minority classes in a dataset to the number of observations of the majority class.
    """
    def __init__(self, x_train, y_train, num_samples=3, contains_categoricals=False, encoded=False, cat_cols=None):
        """
        dataframe: A pandas dataframe that containes the features and target
        target_column: Target column name as a string or target column index as an integer
        num_samples: Default to 3. During the oversampling process, the num_samples will determine the
            number of points used to generate new feature values for a new point
        contains_categoricals: Default to False. If the dataset contains categorical features, set to True
        encoded: Default to False. If the dataset contains dummy encoded categorical features, set to True
        cat_cols: Default to None. If the dataset contains categorical features, provide a list of the column indices
        """


        self.Data = pd.concat([y_train, x_train], axis=1) #concatenate the target and features into one dataframe
        self.target_ind = 0 #target column index is 0 becasue the target column is the first column in self.Data
        self.num_samples = num_samples
        self.contains_categoricals = contains_categoricals
        self.encoded = encoded
        self.cat_cols = cat_cols
        self.cols = x_train.columns.tolist()
        

    def fit(self):
        """
        This method will iterate through each minority class and artifically create observations with new
        feature values using RF-proximities as weights. This method will return a tuple of:
            1st: A pandas dataframe (x) containing all features and their values (including the new data points)
            2nd: A padnas series (y) containing all target values (including the new data points) corresponding
                to the correct index in x.
        """

        numof_cat_cols = 0
        numof_num_cols = 0

        #make a dictionary with each categorical column and its unique values. This will be useful further down the line
        cat_dict = {}
        if self.contains_categoricals:
            if not self.encoded:
                # If NOT encoded, store unique values for each categorical column
                for col in self.cat_cols:
                    cat_dict[col] = self.Data[col].nunique()
                    numof_cat_cols = len(self.cat_cols)
            elif self.encoded:
                # If encoded, infer categorical groups from one-hot encoded column names
                encoded_groups = {}
                for col in self.Data.columns: # Loop through all columns in the DataFrame to find categoricals
                    for cat_col in self.cat_cols:
                        if col.startswith(cat_col + "_"):  # Check if the column belongs to a one-hot encoded group
                            encoded_groups[cat_col] = encoded_groups.get(cat_col, 0) + 1
                            numof_cat_cols += 1

                cat_dict = encoded_groups  # Store the grouped one-hot column names
        numof_num_cols = len(self.Data.columns) - numof_cat_cols - 1 #calculate number of numerical columns (-1 for target variable)

        x = None
        y = None

        if not self.encoded and self.contains_categoricals:    
            data_encoded = pd.get_dummies(self.Data, columns=self.cat_cols, dtype=int) #encode the categorical columns
            y = data_encoded.iloc[:, self.target_ind]
            x = data_encoded.drop(self.Data.columns[self.target_ind], axis=1)
        else:
            y = self.Data.iloc[:, self.target_ind]
            x = self.Data.drop(self.Data.columns[self.target_ind], axis=1)

        #Train RF, get proximities
        rf = RFGAP(y = y, prediction_type = 'classification', matrix_type = 'dense')
        rf.fit(x, y)
        prox = rf.get_proximities()

        #Get the Majority Class, its count, and store all classes and their counts in a dictionary. This will be useful further down the line
        value_counts = y.value_counts()
        maj_label = value_counts.index[0]
        maj_count = value_counts.iloc[0]
        class_counts = value_counts.to_dict()

        x_numerical = None
        x_categorical = None

        # Separate numerical and categorical data if data contains categorical features
        if self.contains_categoricals:
            encoded_cat_cols = x.columns[numof_num_cols:]
            x_numerical = x.drop(columns=encoded_cat_cols)
            x_categorical = x[encoded_cat_cols]
        else:
            x_numerical = x

        #store number of features in each type
        num_features_size = x_numerical.shape[1]
        cat_features_size = x_categorical.shape[1] if self.contains_categoricals else 0

        #Loop through all classes that aren't majority class and upsample
        for label in class_counts:
            if label != maj_label:

                #Get number of samples to upsample
                #initialize arrays to store new numerical and categorical features
                upsample_size = maj_count - class_counts[label]
                new_points_num = np.zeros((upsample_size, num_features_size))
                new_points_cat = np.zeros((upsample_size, cat_features_size), dtype=int)

                #generate new samples until the number of samples is equal to the majority class
                for i in range(upsample_size):

                    #sample random points from the class that needs to be upsampled
                    sample_indices = np.where(y == label)[0]
                    samples = np.random.choice(sample_indices, self.num_samples, replace=True)

                    #Get new numerical features and add to new points array
                    new_features_num = np.sum((x_numerical.T.iloc[:, samples] @ prox[samples, :]), axis=1)/self.num_samples #proximity weighted average of the samples
                    new_points_num[i, : ] = new_features_num

                    if self.contains_categoricals:
                        #Get new categorical features and add to new points array
                        start_x_slice_ind = 0 #variable to help in slicing categorical columns to focus on one encoded categorical feature at a time
                        for categorical_feature in cat_dict: #loop through the features in the encoded x_categorical columns

                            numof_values_in_feature = cat_dict[categorical_feature] #number of columns in x_categorical dedicated to specified encoded feature - used in slicing
                            end_x_slice_ind = numof_values_in_feature + start_x_slice_ind
                            this_encoded_feature = x_categorical.iloc[:, start_x_slice_ind:end_x_slice_ind] #slice to focus on the encoded columns of a feature, one feature at a time

                            new_feature_cat_ind = np.sum((this_encoded_feature.T.iloc[:, samples] @ prox[samples, :]).T, axis=0).argmax(axis=0) #proximity weighted 'best guess' of the samples

                            #loop through the columns of the encoded feature and set the column with the highest proximity to 1 and the rest to 0
                            for j in range(numof_values_in_feature): 
                                if j == new_feature_cat_ind:
                                        new_points_cat[i, j + start_x_slice_ind] = 1
                                else:
                                        new_points_cat[i, j + start_x_slice_ind] = 0
                            # ✅ Add this right after the inner loop finishes
                            encoded_row = new_points_cat[i, start_x_slice_ind:end_x_slice_ind]
                            if encoded_row.sum() > 1:
                                raise ValueError(f"Multi-assignment detected for {categorical_feature} in synthetic row {i}. Values: {encoded_row}")

                            start_x_slice_ind = end_x_slice_ind #update starting index for slicing

                if self.contains_categoricals:
                    #concatenate the new numerical and categorical features to the original data
                    if not self.encoded:
                        new_points_cat_non_dummy = pd.from_dummies(pd.DataFrame(new_points_cat, columns=x_categorical.columns), sep='_')
                        new_combined_x = pd.concat((pd.DataFrame(new_points_num, columns=x_numerical.columns), 
                                                    new_points_cat_non_dummy), axis=1)

                        old_points_cat_non_dummy = pd.from_dummies(x_categorical, sep='_')
                        old_combined_x = pd.concat((x_numerical, old_points_cat_non_dummy), axis=1)
                    else:
                        new_combined_x = pd.concat((pd.DataFrame(new_points_num, columns=x_numerical.columns), 
                                                    pd.DataFrame(new_points_cat, columns=x_categorical.columns)), axis=1)
                        old_combined_x = pd.concat((x_numerical, 
                                                    x_categorical), axis=1)
                else:
                    old_combined_x = x_numerical
                    new_combined_x = pd.DataFrame(new_points_num, columns=x_numerical.columns)
                x = pd.concat((old_combined_x, 
                            new_combined_x), axis=0).reset_index(drop=True)
                y = pd.concat((y, 
                            pd.Series(np.ones_like(new_points_num[:, 0]) * label, 
                                        dtype=int))).reset_index(drop=True)

        missing_cols = set(self.cols) - set(x.columns)
        for col in missing_cols:
            x[col] = 0

        x = x[self.cols]

        return x, y
