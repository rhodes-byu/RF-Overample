import numpy as np
import pandas as pd
from rfgap import RFGAP


class RFOversample:
    """
    Currently (01-28-25) wrtten to upsample all minority classes in a dataset to the number of observations
    of the majority class. 
    """
    def __init__(self, dataframe, target_column, num_samples=3):
        """
        dataframe: A pandas dataframe that containes the features and target
        target_column: Target column name as a string or target column index as an integer
        num_samples: Default to 3. During the oversampling process, the num_samples will determine the
            number of points used to generate new feature values for a new point
        """
        if isinstance(target_column, str):
            self.y = dataframe[target_column]
            self.x = dataframe.drop([target_column], axis = 1)
        elif isinstance(target_column, int):
            self.y = dataframe.iloc[:,target_column]
            self.x = dataframe.drop([data.columns[target_column]])
        self.num_samples = num_samples
        self.rf = RFGAP(y = self.y, prediction_type = "Classification", matrix_type = 'dense')


    def fit(self):
        """
        This method will iterate through each minority class and artifically create observations with new
        feature values using RF-proximities as weights. This method will return a tuple of:
            1st: A pandas dataframe (x) containing all features and their values (including the new data points)
            2nd: A padnas series (y) containing all target values (including the new data points) corresponding
                to the correct index in x.
        """
        self.rf.fit(self.x, self.y)
        prox = self.rf.get_proximities()

        value_counts = self.y.value_counts()
        maj_label = value_counts.index[0]
        maj_count = value_counts.iloc[0]
        class_counts = value_counts.to_dict()

        num_features = self.x.shape[1]

        for label in class_counts:
            if label != maj_label:
                upsample_size = maj_count - class_counts[label]
                new_points = np.zeros((upsample_size, num_features))

                for i in range(upsample_size):
                    sample_indices = np.where(self.y == label)[0]
                    samples = np.random.choice(sample_indices, self.num_samples, replace=True)
                    new_features = np.sum((np.matmul(np.transpose(self.x.iloc[samples, :]), prox[samples, :])), axis=1) / self.num_samples
                    new_points[i, :] = new_features

                self.x = pd.concat((self.x, pd.DataFrame(new_points, columns=self.x.columns))).reset_index(drop=True)
                self.y = pd.concat((self.y, pd.Series(np.ones_like(newPoints[:, 0]) * label))).reset_index(drop=True)

        return self.x, self.y


