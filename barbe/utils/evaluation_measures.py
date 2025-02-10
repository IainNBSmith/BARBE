import numpy as np
import pandas as pd


class EuclideanDistanceInterval:

    def __init__(self):
        # can use euclidean distance markers (<1: 10 (value given), ...) = {1: 10, ...}
        # can create distance markers with training data (consider the distribution of the data)
        # hamming distance for the categorical values? well maybe
        # could use cosine similarity
        aa = 1

    def get_nearest_neighbor_distance(self, reference_data, input_data, interval_measures=None, full_detail=False):
        if interval_measures is None:
            interval_measures = {30: 80,  # % of data: % of value
                                 90: 15,  # i.e. the closest 50% of data is 80% of the weight
                                 100: 5}  # with 200 records the 100 closest each contribute 0.8% if correct
        distance_list = np.linalg.norm(reference_data.to_numpy() -
                                       input_data.to_numpy(),
                                       axis=1)
        weights = np.zeros(shape=(len(distance_list), 1))
        previous_cutoff = -1e-16
        for percentile in interval_measures.keys():
            percentage_contribution = interval_measures[percentile]
            cutoff = np.percentile(distance_list, percentile)
            index_cutoff = np.where(previous_cutoff < distance_list)[0]
            index_cutoff = index_cutoff[np.where(distance_list[index_cutoff] <= cutoff)]
            individual_contribution = percentage_contribution/(100*len(index_cutoff))
            weights[index_cutoff] = individual_contribution
            previous_cutoff = cutoff

        if full_detail:
            return weights, distance_list
        return weights

    def get_euclidean_distance(self, reference_data, input_data):

        distance_list = np.linalg.norm(reference_data.to_numpy() -
                                       input_data.to_numpy(),
                                       axis=1)
        total_distance = np.nansum(distance_list)
        weights = total_distance / (distance_list + 1e-6)
        weights = weights / np.nansum(np.ma.masked_invalid(weights))
        return weights


from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler


def _get_euclidean(reference_data, input_data, scaled=True):
    if isinstance(reference_data, pd.Series):
        reference_data = reference_data.to_numpy().reshape(1, -1)
    if isinstance(input_data, pd.DataFrame):
        input_data.to_numpy()
    #print(input_data.shape)
    #print(reference_data.shape)
    if scaled:
        temp_scaler = StandardScaler()
        input_data = temp_scaler.fit_transform(input_data)
        reference_data = temp_scaler.transform(reference_data)[0]
        #print(reference_data)

    reference_data = np.nan_to_num(reference_data, nan=0)
    input_data = np.nan_to_num(input_data, nan=0)
    all_euclideans = []
    for i in range(input_data.shape[0]):
        #print(reference_data)
        #print(input_data[i])
        all_euclideans.append(euclidean(reference_data, input_data[i]))
    return np.array(all_euclideans)


def nearest_neighbor_weights(reference_data, input_data, interval_measures=None, full_detail=False):
    if interval_measures is None:
        interval_measures = {20: 100,  # % of data: % of value contribution to weight
                             100: 0}  # i.e. the closest 20% of data is 80% of the weight
                             #100: 5}  # with 2000 records the 100 closest each contribute 0.8% if correct

    #if input_data.shape[1] != len(reference_data):
    #    input_data = input_data.T
    distance_list = _get_euclidean(reference_data, input_data)
    #print(reference_data)
    #print(distance_list)
    weights = np.zeros(len(distance_list))
    previous_cutoff = -1e-16
    for percentile in interval_measures.keys():
        percentage_contribution = interval_measures[percentile]
        cutoff = np.percentile(distance_list, percentile)
        index_cutoff = np.where(previous_cutoff < distance_list)[0]
        index_cutoff = index_cutoff[np.where(distance_list[index_cutoff] <= cutoff)]
        #print("LEN INDEX: ", len(index_cutoff))
        individual_contribution = percentage_contribution/(100*len(index_cutoff))
        weights[index_cutoff] = individual_contribution
        previous_cutoff = cutoff

    if full_detail:
        return weights, distance_list
    #print(weights)
    return weights


def euclidean_weights(reference_data, input_data):
    #if input_data.shape[0] != len(reference_data):
    #    input_data = input_data.T

    distance_list = _get_euclidean(reference_data, input_data)
    #print(reference_data)
    #print(distance_list)
    total_distance = np.nansum(distance_list)
    weights = total_distance / (distance_list+1e-8)
    weights = weights / np.nansum(np.ma.masked_invalid(weights))
    return weights
