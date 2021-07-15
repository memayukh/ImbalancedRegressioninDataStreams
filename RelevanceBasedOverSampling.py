# importing required modules
import numpy as np


def RelevanceOverSampling(D, target, size, relevance, categorical_col = []):
    """

    :param D: Data in the form of Dataframe
    :param target: target variable in the data
    :param size: Number of duplicate rare events to generate
    :param relevance: Relevance Phi values for continous target variable
    :param categorical_col: list of column names that are categorical in nature.
    :return: Resampled Data
    """

    # If all relevance values are zero
    if(int(sum(relevance)) == 0):
        return D

    # Normalizing the relevance values
    relevance = [float(i)/sum(relevance) for i in relevance]
    # Returns the size number of duplicate values representing rare events
    new_target_values = np.random.choice(D[target], size, p=relevance)

    new_indices = []
    for y_instance in new_target_values:
        index = D.index[D[target] == y_instance].tolist()
        new_indices.append(index[-1])


    extra_data = D.loc[new_indices]

    new_D = D.append(extra_data, ignore_index = True)

    return new_D









