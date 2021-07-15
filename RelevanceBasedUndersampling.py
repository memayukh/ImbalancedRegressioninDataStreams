# importing required modules
import numpy as np


def RelevanceUnderSampling(D, target, size, relevance, categorical_col = []):
    """

    :param D: Data in the form of Dataframe
    :param target: target variable in the data
    :param size: Number of samples to be removed
    :param relevance: Relevance Phi values for continous target variable
    :param categorical_col: list of column names that are categorical in nature.
    :return: Resampled Data
    """

    # If all relevance values are zero
    if int(sum(relevance))==0:
        return D

    L = len(D) #Length of the data frame
    relevance = [float(i)/sum(relevance) for i in relevance] #normalizing the relevance values



    # D - dataframe ---- D.index.values.tolist() ->>> list of indices

    final_size = L-size
    # This is returning the size number of duplicate values representing the rare cases
    new_target_values = np.random.choice(D.index.values, final_size, p=relevance)
    ntv = new_target_values.reshape(-1,1)


    for y_instance in ntv:
        new_D = D.loc[y_instance]









    return new_D
