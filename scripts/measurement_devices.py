#
# This file is part of the autonoexp distribution
# (https://gitlab.ruhr-uni-bochum.de/stricm9y/pyiron_exp_utils)
# Copyright (c) 2022 Markus Stricker.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""Class mimicking an experimental measurement device"""

import numpy as np


# todo: add x/y coordinates on waver
class Resistance:
    def __init__(self, df_measurement, features=None, target=None):
        """Initialize device with existing pandas dataframe form"""
        self.df = df_measurement
        self.features = features
        self.target = target
        print("Headers of DataFrame:\n", self.df.columns.values)

        # Initialize for keeping track of to-the-outside-known-values
        self.measured_ids = []

    def get_features(self):
        """Returns the variables for a Gaussian Process"""
        return self.df[self.features].values

    def get_data(self):
        """Returns all known variables from the file"""
        return self.raw_df

    def register_measure(self, indices):
        """Keeping track of already measured points by storing their index

        Arguments:
        ----------
        indices -- list of integers

        """
        for index in indices:
            if index not in self.measured_ids:
                self.measured_ids.append(index)
            # else:
            #     print(
            #         "ID to be measured: {}, measured IDs: {}".format(
            #             index, self.measured_ids
            #         )
            #     )
            #     raise ValueError("Measurement index already registered.")

    def get_initial_measurement(self, indices=None, target_property="Resistance"):
        """Provides an initial measurement
        Arguments:
        ----------
        indices -- (list of integers) if not None, the respective
                   measurement entries are returned, otherwise a random
                   choice of 5 is returned

        Returns:
        --------
        A chosen or random subset of the measurement with

        X -- element concentrations

        y -- measured resistance
        """

        if indices is None:
            nchoice = 5
            indices = np.random.choice(np.arange(self.size), size=nchoice)
            print("Random indices = {}".format(indices))

        return self.get_measurement(indices, target_property=target_property)

    def get_measurement(self, indices, target_property="Resistance"):
        """Routine for getting a measurement

        Arguments:
        ----------
        indices -- (list of integers) if not None, the respective
                   measurement entries are returned, otherwise a random
                   choice of 5 is returned

        Returns:
        --------
        A chosen or random subset of the measurement with

        X -- element concentrations

        y -- measured resistance
        """

        # register measured indices
        self.register_measure(indices)

        # return all registered measurements
        X = self.df[self.features].values[self.measured_ids, :]
        y = self.df[target_property].values[self.measured_ids][:, None]

        return X, y
    def get_unmeasured_features(self):
        all_indices = set(range(len(self.df)))
        unmeasured_indices = list(all_indices - set(self.measured_ids))
        return self.df.iloc[unmeasured_indices][self.features].values
    
    def get_index_from_feature(self, feature_row):
        """
        Given a feature row (e.g., from GP's suggestion), find its index in the full dataset
        that has NOT been measured yet.
        """
        all_features = self.df[self.features].values
        # Find all rows that exactly match the feature_row
        matches = np.where(np.all(np.isclose(all_features, feature_row, atol=1e-8), axis=1))[0]
        
        for idx in matches:
            if idx not in self.measured_ids:
                return idx

        raise ValueError("Feature row already measured or not found in the dataset.")

