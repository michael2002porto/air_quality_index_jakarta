import sys
import pandas as pd
import numpy as np

from imblearn.pipeline import Pipeline

import matplotlib.pyplot as plt

from collections import Counter
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

class Preprocessor():
    def __init__(self):
        super(Preprocessor, self).__init__()
        self.preprocessor()
        
    def load_data(self):
        dataset = pd.read_csv("datasets/ispu_dki1.csv")
        dataset = dataset[["pm10", "so2", "co", "o3", "no2", "categori"]]

        return dataset

    def under_sampling_sedang(self, x, y):
        # Under-sampling majority "SEDANG" -> 1000 data (change n_neighbors)
        enn = EditedNearestNeighbours(
            sampling_strategy='majority',
            n_neighbors=170,
            kind_sel='all',
            n_jobs=None
        )
        x_resampled, y_resampled = enn.fit_resample(x, y)
        print('Resampled dataset shape (ENN) %s' % sorted(Counter(y_resampled).items()))
        
        return x_resampled, y_resampled

    def hybridization_tidak_sehat(self, x_resampled, y_resampled):
        # sme = SMOTEENN(
        #     sampling_strategy='minority',
        #     random_state=None,
        #     smote=None,
        #     enn=None,
        #     n_jobs=None
        # )
        # x_resampled, y_resampled = sme.fit_resample(x_resampled, y_resampled)
        # print('Resampled dataset shape (SMOTENN) %s' % Counter(y_resampled))

        # Over-sampling set manual "TIDAK SEHAT" -> 2800 data (change sampling_strategy)
        smote = SMOTE(
            sampling_strategy={'TIDAK SEHAT': 2800},
            random_state=None,
            k_neighbors=5,
            n_jobs=None
        )
        x_resampled, y_resampled = smote.fit_resample(x_resampled, y_resampled)
        print('Resampled dataset shape (SMOTE) %s' % sorted(Counter(y_resampled).items()))

        # Under-sampling majority "TIDAK SEHAT" -> over 1000 data (change n_neighbors)
        enn = EditedNearestNeighbours(
            sampling_strategy='majority',
            n_neighbors=1950,
            kind_sel='all',
            n_jobs=None
        )
        x_resampled, y_resampled = enn.fit_resample(x_resampled, y_resampled)
        print('Resampled dataset shape (ENN) %s' % sorted(Counter(y_resampled).items()))

        return x_resampled, y_resampled
    
    def preprocessor(self):
        dataset = self.load_data()

        # Filter out rows where the 'categori' column is "TIDAK ADA DATA"
        dataset = dataset[dataset['categori'] != 'TIDAK ADA DATA']

        # SEDANG = 3065
        # BAIK = 1054
        # TIDAK SEHAT = 154

        # Let’s see the data description and check missing values
        print(dataset.info())
        print(dataset.isnull().sum())
        print('hello')

        # Removing Rows with Missing Values
        dataset = dataset.dropna()
        print(dataset.info())
        print(dataset.isnull().sum())

        # SEDANG = 2877
        # BAIK = 928
        # TIDAK SEHAT = 145

        # Count the occurrences of each class in 'categori'
        category_count = dataset['categori'].value_counts()
        print("\nCount of data for each class in 'categori':")
        print(category_count)
        print("\n")

        # Save the Removing Rows with Missing Values to a CSV file
        preprocessed_dataset = pd.DataFrame(dataset)
        preprocessed_dataset.to_csv('preprocessed_ispu_dki1.csv', index=False)

        #Split the Features (X) and Target (Y)
        y = dataset['categori'].values
        x = dataset.drop('categori', axis=1)

        x_res, y_res = self.under_sampling_sedang(x, y)
        x_res, y_res = self.hybridization_tidak_sehat(x_res, y_res)

        # sys.exit()

        # If x_res is a numpy array, convert it to a DataFrame
        if not isinstance(x_res, pd.DataFrame):
            x_res = pd.DataFrame(x_res)

        # Convert y_res to a Series if it's not already
        if not isinstance(y_res, pd.Series):
            y_res = pd.Series(y_res)

        # Convert all values to integers
        x_res = x_res.astype(int)

        # Add y_res as a new column to x_res
        x_res['categori'] = y_res

        # Save the combined DataFrame to a CSV file
        x_res.to_csv('resampled_ispu_dki1.csv', index=False)

        return x_res, y_res