import numpy as np
import pandas as pd
import copy

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


class OneHotEncoderDF(BaseEstimator, TransformerMixin):
    """
    A version of OneHotEncoder that return a new Dataframe with applied OneHotEncoder for selected columns and
    automatically renames columns' names.
    Warning!
    Encoded columns will be moved to the start of the DataFrame.

    """

    def __init__(
            self,
            columns,
            categories="auto",
            drop=None,
            sparse_output=None,
            dtype=np.int32,
            handle_unknown="error",
    ):
        """
            Parameters:
                columns: list of columns for transformation
                onehotencoders_: list of OneHotEncoders fot each selected column
                column_names_: list of new columns' names after encoding
                rest: according to the documentation
                    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        """
        self.columns = columns
        self.onehotencoders_ = []
        self.column_names_ = []
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        pass

    def fit(self, X, y=None):
        """To each selected column fit a separate OneHotEncoder.

            Parameters:
                X: DataFrame
                y: None, ignored. This parameter exists only for compatibility with pipeline

            Returns:
                self: object
                    Fitted encoder

            Raises:
                TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        for c in self.columns:
            # constructs the OHE parameters
            ohe_params = {
                "categories": self.categories,
                "drop": self.drop,
                "sparse_output": False,
                "dtype": self.dtype,
                "handle_unknown": self.handle_unknown,
            }

            ohe = OneHotEncoder(**ohe_params)
            self.onehotencoders_.append(ohe.fit(X.loc[:, [c]]))

            # returns columns names from OneHotEncoder
            feature_names = ohe.get_feature_names_out()
            self.column_names_.append(feature_names)

        # fit method returns the transformer itself
        return self

    def transform(self, X):
        """Transform given DataFrame X using the OneHotEncoder for selected columns.

            Parameters:
                X: DataFrame

            Returns:
                New Dataframe where selected columns have been transformed

            Raises:
                NotFittedError if the transformer is not yet fitted
                TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        if not hasattr(self, 'onehotencoders_'):
            raise NotFittedError(f"{type(self).__name__} is not fitted")

        all_df = []  # list with Dataframes

        for i, c in enumerate(self.columns):
            # fits transformer for selected column
            ohe = self.onehotencoders_[i]
            # transforms selected column
            transformed_col = ohe.transform(X.loc[:, [c]])
            # constructs DataFrame where each row represents one class from the original column
            df_col = pd.DataFrame(transformed_col, columns=self.column_names_[i])
            # adds DataFrame to list
            all_df.append(df_col)
        # makes a copy original DataFrame without transformed columns
        result = X.drop(self.columns, axis=1)
        # adds to the tail of list
        all_df.append(result)

        # returns a NEW Dataframe with transformed features
        return pd.concat(all_df, axis=1)


class OrdinalEncoderDF(BaseEstimator, TransformerMixin):
    """Integer encoding consists in replacing the categories by digits from 1 to n (or 0 to n-1,
       depending on the implementation), where n is the number of distinct categories of the variable.
       The numbers are assigned arbitrarily.

    """

    def __init__(
            self,
            columns,
            categories="auto",
            dtype=np.int32,
            handle_unknown="error",
    ):
        """
            Parameters:
                columns: list of columns for transform
                rest: according to the documentation
                    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html

        """
        self.columns = columns
        self.labels_ = None
        self.labelencoder_ = None
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit a OrdinalEncoder with selected columns

            Parameters:
                X: dataframe
                y: None, ignored. This parameter exists only for compatibility with Pipeline

            Returns:
                self

            Raises:
                TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        ordinal_params = {
            "categories": self.categories,
            "dtype": self.dtype,
            "handle_unknown": self.handle_unknown,
        }

        self.labelencoder_ = OrdinalEncoder(**ordinal_params)
        self.labelencoder_.fit(X[self.columns])

        # returns mapping dictionary from LabelEncoder
        encoding = self.labelencoder_.categories_
        # creates a dictionary with mapping
        encoding_feature = lambda x: dict(zip(x, range(len(x))))
        # creates a list with mapping dictionaries
        encoding_full = [encoding_feature(feature_elem) for feature_elem in encoding]

        # stores a dictionary with mappings for each column
        self.labels_ = {self.columns[i]: encoding_full[i] for i in range(len(self.columns))}

        # fit method returns the transformer itself
        return self

    def transform(self, X):
        """Create new DataFrame with transformed columns

            Parameters:
                X: Dataframe

            Returns:
                New Dataframe where selected columns have been transformed

            Raises
                NotFittedError if the transformer is not yet fitted
                TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        if not hasattr(self, "labelencoder_"):
            raise NotFittedError(f"{type(self).__name__} is not fitted")

        # creates a copy of a DataFrame
        X_copy = X.copy()

        # applies OrdinalEncoder (creates a Dataframe with transformed columns)
        X_trans = self.labelencoder_.transform(X_copy[self.columns])

        # replaces transformed columns in original DataFrame
        for i, c in enumerate(self.columns):
            X_copy[c] = X_trans[:, i]

        return X_copy

    def get_transform_dic(self):
        """"Returns a dictionary with all mappings
                key: feature name
                value: mapping dictionary
        """
        return self.labels_


class CountOfFreqEncoder(BaseEstimator, TransformerMixin):
    """Replaces the categories by the count of the observations that show that category in the dataset.
    Similarly, the category can be replaced with the frequency -or percentage of observations in the dataset.
    """

    def __init__(
            self,
            columns,
    ):
        """
            Parameters:
                columns: list of columns for transform

        """
        self.labels_ = None
        self.mapping = []
        self.columns = columns

    def fit(self, X, y=None):
        """Creates a mapping dictionary for transformation.
           Values for replacement are calculated based on the frequency of each class in the DataFrame

            Parameters:
                X: dataframe
                y: None, ignored.

            Returns:
                self

            Raises:
                TypeError if X is not of type DataFrame
        """
        
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        for col in self.columns:
            # calculates frequency of each class
            frequency_map = (X[col].value_counts() / len(X)).to_dict()
            # adds to mapping list
            self.mapping.append(frequency_map)

        # stores mapping dictionary as an attribute
        self.labels_ = {self.columns[i]: self.mapping[i] for i in range(len(self.columns))}

        # fit method should return the transformer itself
        return self

    def transform(self, X):
        """Transform selected columns with self.mapping

            Parameters:
                X: original Dataframe

            Returns:
                New Dataframe where selected columns have been encoded

            Raises:
                NotFittedError if the transformer is not yet fitted
                TypeError if X is not of type DataFrame
        """

        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        if not hasattr(self, "mapping"):
            raise NotFittedError(f"{type(self).__name__} is not fitted")

        X_copy = X.copy()
        for i, c in enumerate(self.columns):
            X_copy[c] = X_copy[c].map(self.mapping[i])

        return X_copy

    def get_transform_dic(self):
        return self.labels_


class OrderedIntTargetEncoder(BaseEstimator, TransformerMixin):
    """ Ordering the categories according to the target means assigning a number
        to the category from 1 to k, where k is the number of distinct categories
        in the variable, but this numbering is informed by the mean of the target
        for each category.

    """

    def __init__(
            self,
            columns,
    ):
        """
        Parameters:
            columns: list of columns to transform

        """
        self.labels_ = None
        self.columns = columns
        self.mapping = []

    def fit(self, X, y):
        """ Creates a mapping dictionary for transformation.
            Values for replacement are calculated based on the mean target value.
                Example:
                    For the variable color, if the mean of the target for blue, red, and grey is 0.5, 0.8 and 0.1
                    respectively, blue is replaced by 1, red by 2 and grey by 0.

            Parameters:
                X: dataframe
                y: target

            Returns:
                self

            Raises:
                TypeError if X is not of type DataFrame

        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        # adds target to dataframe
        X['target'] = y

        for col in self.columns:
            # creates an ordered list with the labels
            ord_label = X.groupby(col)['target'].mean().sort_values().index
            # creates a dictionary with the mappings of categories to numbers
            ordinal_mapping = {k: i for i, k in enumerate(ord_label, 0)}
            # adds mapping to final list
            self.mapping.append(ordinal_mapping)

        # stores mapping dictionary as an attribute
        self.labels_ = {self.columns[i]: self.mapping[i] for i in range(len(self.columns))}

        # drops target
        X.drop('target', inplace=True, axis=1)

        # fit method return the transformer itself
        return self

    def transform(self, X):
        """Transform selected columns with self.mapping

        Parameters:
            X: Dataframe that is to be one hot encoded

        Returns:
            New Dataframe where selected columns have been encoded

        Raises
            NotFittedError if the transformer is not yet fitted
            TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        if not hasattr(self, "mapping"):
            raise NotFittedError(f"{type(self).__name__} is not fitted")

        # makes a copy of original FataFrame
        X_copy = X.copy()
        for i, c in enumerate(self.columns):
            # replaces the labels with the integers
            X_copy[c] = X_copy[c].map(self.mapping[i])

        return X_copy

    def get_transform_dic(self):
        """"Returns a dictionary with all mappings
                key: feature name
                value: mapping dictionary
        """

        return self.labels_


class AggTargetEncoder(BaseEstimator, TransformerMixin):
    """ Agg encoding implies replacing the category with the aggregated target value for that category.
    """

    def __init__(
            self,
            columns,
            agg_fun
    ):
        """
        Parameters:
            columns: list of columns for transform
            agg_fun: aggregating functions for target.
                     List of functions:
                        mean(): Compute mean
                        sum(): Compute sum
                        size(): Compute group sizes
                        count(): Compute count
                        std(): Standard deviation
                        var(): Compute variance
                        sem(): Standard error of the mean
                        first(): Compute first of group values
                        last(): Compute last of group values
                        nth() : Take nth value, or a subset if n is a list
                        min(): Compute min
                        max(): Compute max

        """
        self.columns = columns
        self.agg_fun = agg_fun
        self.mapping_ = []
        self.labels_ = None

    def fit(self, X, y):
        """Creates a mapping dictionary for transformation.
            Values for replacement are calculated based on the given agg function and target.

            Parameters:
                X: dataframe
                y: target

            Returns:
                self

            Raises:
                TypeError if X is not of type DataFrame
                AttributeError if agg_fun can't be applied
            """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        agg_list = ['mean',
                    'sum',
                    'size',
                    'count',
                    'std',
                    'var',
                    'sem',
                    'first',
                    'last',
                    'nth',
                    'min',
                    'max']

        if self.agg_fun not in agg_list:
            raise AttributeError(f"'SeriesGroupBy' object has no attribute {self.agg_fun}")

        X['target'] = y

        for col in self.columns:
            m_label = X.groupby(col)['target'].agg(self.agg_fun).to_dict()
            self.mapping_.append(m_label)

        # stores mapping dictionary as an attribute
        self.labels_ = {self.columns[i]: self.mapping_[i] for i in range(len(self.columns))}

        X.drop('target', inplace=True, axis=1)

        # fit method should return the transformer itself
        return self

    def transform(self, X):
        """Transform selected columns with self.mapping

            Parameters:
                X: Dataframe that is to be one hot encoded

            Returns:
                New Dataframe where selected columns have been encoded

            Raises:
                NotFittedError if the transformer is not yet fitted
                TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        if not hasattr(self, "mapping_"):
            raise NotFittedError(f"{type(self).__name__} is not fitted")

        X_copy = X.copy()
        for i, c in enumerate(self.columns):
            X_copy[c] = X_copy[c].map(self.mapping_[i])

        return X_copy

    def get_transform_dic(self):
        """"Returns a dictionary with all mappings
                key: feature name
                value: mapping dictionary
        """
        return self.labels_

