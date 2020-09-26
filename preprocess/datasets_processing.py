import random
from typing import List

from model.EpitopesClusters import EpitopesClusters
from model.EpitopesDataset import EpitopesDataset


def get_epitopes_with_max_verified_regions(epitopes_clusters: EpitopesClusters) -> EpitopesDataset:
    """
    Gets a dataset of the epitope records with maximum verified regions from each cluster
    Parameters
    ----------
    epitopes_clusters: model.EpitopesDataset.EpitopesDataset

    Returns
    -------
    max_verified_regions_epitopes_dataset : model.EpitopesDataset.EpitopesDataset
        Dataset of the epitope records with maximum verified regions from each cluster
    """
    remaining_epitopes = []

    for cluster in epitopes_clusters:
        epitope_with_max_verified_regions = cluster[0]
        max_verified_regions = len(cluster[0].verified_regions)

        for epitope in cluster:
            if len(epitope.verified_regions) > max_verified_regions:
                epitope_with_max_verified_regions = epitope
                max_verified_regions = len(epitope.verified_regions)

        remaining_epitopes.append(epitope_with_max_verified_regions)

    return EpitopesDataset(remaining_epitopes)


def split_epitopes_clusters_to_cv_datasets(
        epitopes_clusters: EpitopesClusters,
        cv_fold: int,
        shuffle_clusters=True) -> List[EpitopesDataset]:
    """
    Splitting the epitopes records to CV datasets

    Parameters
    ----------
    epitopes_clusters : model.EpitopesDataset.EpitopesDataset

    cv_fold : int
    shuffle_clusters : bool

    Returns
    -------
    epitopes_cv_datasets : List[model.EpitopesDataset.EpitopesDataset]
        CV epitope records datasets
    """
    epitopes_cv_groups = [[]]

    set_approximate_size = - (- epitopes_clusters.get_num_of_epitopes() // cv_fold)

    if shuffle_clusters:
        epitopes_clusters = list(epitopes_clusters)
        random.shuffle(epitopes_clusters)

    curr_groups_ind = 0
    for cluster in epitopes_clusters:
        epitopes_cv_groups[curr_groups_ind].extend(cluster)
        if len(epitopes_cv_groups[curr_groups_ind]) >= set_approximate_size:
            epitopes_cv_groups.append([])
            curr_groups_ind += 1

    if len(epitopes_cv_groups[-1]) == 0:
        epitopes_cv_groups = epitopes_cv_groups[:-1]


    epitopes_cv_datasets = [EpitopesDataset(epitopes_group) for epitopes_group in epitopes_cv_groups]

    return epitopes_cv_datasets
