import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsRegressor
from itertools import combinations


def _handle_value_error(new_cases, D, ng, target, categorical_col):
    df_len = D.shape[0]  # length of dataframe or total number of instances

    if df_len == 1:
        # if there is only one instance, then we replicate it
        for _ in range(ng):
            new_cases = new_cases.append(D.iloc[0], ignore_index=True)

    if df_len in [2, 3]:
        # if there are 2 or 3 instances then this logic is followed
        for _ in range(df_len):

            li = []
            c = combinations([i for i in range(df_len)], 2)
            for i in list(c):
                li.append(i)

            # li is the list of possible combinations of indices
            # for e.g. if the df_len is 3, we have 3 indices (0,1 and 2)
            # but we select random 2 instances out of three -> either (0,1)(1,2)(0,2)

            selected_indices = li[np.random.randint(len(li))]

            case = D.iloc[selected_indices[0]]
            x = D.iloc[selected_indices[1]]

            # same logic as Durga's code below
            for _ in range(ng):
                attr = {}

                for a in D.columns:
                    # skip target column
                    if a in [target]:
                        continue

                    if a in categorical_col:
                        # if categorical then choose randomly one of values
                        if np.random.randint(2) == 0:
                            attr[a] = case[a]
                        else:
                            attr[a] = x[a]

                    else:
                        # if continuous column
                        diff = case[a] - x[a]
                        attr[a] = case[a] + np.random.randint(2) * diff

                # decide the target column
                new = np.array(list(attr.values()))

                d1 = cosine_similarity(new.reshape(1, -1), case.drop(labels=[target]).values.reshape(1, -1))[0][0]
                d2 = cosine_similarity(new.reshape(1, -1), x.drop(labels=[target]).values.reshape(1, -1))[0][0]
                attr[target] = (d2 * case[target] + d1 * x[target]) / (d1 + d2)

                # append the result
                new_cases = new_cases.append(attr, ignore_index=True)

    return new_cases


def get_synth_cases(D, target, o=200, k=3, categorical_col=[]):
    '''
    Function to generate the new cases.
    INPUT:
        D - pd.DataFrame with the initial data
        target - string name of the target column in the dataset
        o - oversampling rate
        k - number of nearest neighbors to use for the generation
        categorical_col - list of categorical column names
    OUTPUT:
        new_cases - pd.DataFrame containing new generated cases
    '''
    new_cases = pd.DataFrame(columns=D.columns)  # initialize the list of new cases
    ng = o // 100  # the number of new cases to generate

    # new logic added so as to handle the value error that was occurring in previous code
    new_cases = _handle_value_error(new_cases, D, ng, target, categorical_col)
    if not new_cases.empty:
        # if the _handle_value_error would return new_cases we will return them
        return new_cases

    for index, case in D.iterrows():
        # find k nearest neighbors of the case

        import re
        non_X = [target]
        for cols in D.columns:

            if type(D[cols].iloc[0]).__name__ == 'str' and re.match(r'^-?\d+(?:\.\d+)?$', D[cols].iloc[0]) is None:
                non_X.append(cols)
        knn = KNeighborsRegressor(n_neighbors = k+1) # k+1 because the case is the nearest neighbor to itself
        knn.fit(D.drop(columns = non_X).values, D[[target]])

        k_neighbours_temp = case.drop(labels = non_X).values.reshape(1, -1)
        neighbors = knn.kneighbors(k_neighbours_temp, return_distance=False).reshape(-1)
        neighbors = np.delete(neighbors, np.where(neighbors == index))

        for i in range(0, ng):
            # randomly choose one of the neighbors
            x = D.iloc[neighbors[np.random.randint(k)]]
            attr = {}
            for a in D.columns:
                # skip target column
                if a in non_X:
                    continue;
                if a in categorical_col:
                    # if categorical then choose randomly one of values
                    if np.random.randint(2) == 0:
                        attr[a] = case[a]
                    else:
                        attr[a] = x[a]
                else:
                    # if continious column
                    diff = case[a] - x[a]
                    attr[a] = case[a] + np.random.randint(2) * diff
            # decide the target column
            new = np.array(list(attr.values()))

            d1 = cosine_similarity(new.reshape(1, -1), case.drop(labels=non_X).values.reshape(1, -1))[0][0]
            d2 = cosine_similarity(new.reshape(1, -1), x.drop(labels=non_X).values.reshape(1, -1))[0][0]
            attr[target] = (d2 * case[target] + d1 * x[target]) / (d1 + d2)

            # append the result
            new_cases = new_cases.append(attr, ignore_index=True)

        return new_cases


def SmoteR(D, target, relevance,th = 0.75, o = 200, u = 100, k = 3, categorical_col = []):
    '''
    The implementation of SmoteR algorithm:
    https ://core.ac.uk/download/pdf/29202178.pdf
    INPUT:
        D - pd.DataFrame - the initial dataset
        target - the name of the target column in the dataset
        relevance - Phi values of target column in the form of dataframe or numpy array
        th - relevance threshold
        o - oversampling rate
        u - undersampling rate
        k - the number of nearest neighbors
    OUTPUT:
        new_D - the resulting new dataset
    '''
    # median of the target variable
    y_bar = D[target].median()

    # find rare cases where target less than median
    rareL = D[(relevance > th) & (D[target] > y_bar)]

    # find rare cases where target greater than median
    rareH = D[(relevance > th) & (D[target] < y_bar)]
    # generate rare cases for rareH

    if rareH.empty and rareL.empty:
        return D

    # generate rare cases for rareL
    new_casesL = get_synth_cases(rareL, target, o, k, categorical_col)

    new_casesH = get_synth_cases(rareH, target, o, k, categorical_col)

    new_cases = pd.concat([new_casesL, new_casesH], axis=0)

    # under sample norm cases
    norm_cases = D[relevance <= th]
    # get the number of norm cases
    nr_norm = int(len(norm_cases) * u / 100)

    norm_cases = norm_cases.sample(min(len(D[relevance <= th]), nr_norm))

    # get the resulting dataset
    new_D = pd.concat([new_cases, norm_cases], axis=0)

    return new_D

