import random
from typing import Tuple, List

from model.Epitope import Epitope
from model.EpitopesClusters import EpitopesClusters
from model.EpitopesDataset import EpitopesDataset


def _are_verified_regions_adjacent(verified_region1: Tuple[int, int], verified_region2: Tuple[int, int]) -> bool:
    start1, end1 = verified_region1
    start2, end2 = verified_region2

    return start2 <= start1 <= end2 + 1 or start2 - 1 <= end1 <= end2


def _are_adjacent_verified_regions_in_epitope(epitope: Epitope) -> bool:
    for i in range(len(epitope.verified_regions) - 1):
        for j in range(i + 1, len(epitope.verified_regions)):
            if _are_verified_regions_adjacent(epitope.verified_regions[i], epitope.verified_regions[j]):
                return True

    return False


# todo: add to doc - should run after removing subsets
def _count_adjacent_verified_regions(epitope: Epitope) -> int:
    adjacent_verified_regions = set()

    for i in range(len(epitope.verified_regions) - 1):
        for j in range(i + 1, len(epitope.verified_regions)):
            if _are_verified_regions_adjacent(epitope.verified_regions[i], epitope.verified_regions[j]):
                adjacent_verified_regions.add(epitope.verified_regions[i])
                adjacent_verified_regions.add(epitope.verified_regions[j])

    return len(adjacent_verified_regions)


def count_records_with_adjacent_verified_regions(epitope_dataset: EpitopesDataset) -> int:
    epitopes_with_adjacent_verified_regions_count = 0

    for epitope in epitope_dataset:
        if _are_adjacent_verified_regions_in_epitope(epitope):
            epitopes_with_adjacent_verified_regions_count += 1

    return epitopes_with_adjacent_verified_regions_count


def count_total_adjacent_verified_regions(epitope_dataset: EpitopesDataset) -> int:
    total_adjacent_verified_regions_count = 0

    for epitope in epitope_dataset:
        total_adjacent_verified_regions_count += _count_adjacent_verified_regions(epitope)

    return total_adjacent_verified_regions_count


def count_verified_regions_in_records_with_adjacent_verified_regions(epitopes_dataset: EpitopesDataset) -> int:
    verified_regions_in_records_with_adjacent_verified_regions_count = 0

    for epitope in epitopes_dataset:
        if _are_adjacent_verified_regions_in_epitope(epitope):
            verified_regions_in_records_with_adjacent_verified_regions_count += len(epitope.verified_regions)

    return verified_regions_in_records_with_adjacent_verified_regions_count


def get_epitopes_with_max_verified_regions(epitopes_clusters: EpitopesClusters) -> EpitopesDataset:
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
