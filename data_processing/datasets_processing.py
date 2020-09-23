from typing import Tuple

from models.Epitope import Epitope
from models.EpitopesDataset import EpitopesDataset
from models.EpitopesClusters import EpitopesClusters



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
