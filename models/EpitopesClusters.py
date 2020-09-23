CLUSTER_PREFIX = '>'
EPITOPE_ID_PREFIX = '>'
EPITOPE_ID_SUFFIX = '...'


class EpitopesClusters:
    def __init__(self, clstr_file_path: str):
        self.__epitopes_clusters_lst = self.__parse_clstr_file(clstr_file_path)

    @staticmethod
    def __parse_clstr_file(clstr_file_path: str):
        epitopes_clusters_lst = []

        with open(clstr_file_path) as epitopes_ids_clusters_file:
            curr_cluster = []
            for line in epitopes_ids_clusters_file.readlines():
                line = line.strip()
                # when new cluster found appending the current cluster set and creating new one
                # if the cluster set is not empty (should occur on first line)
                if line.startswith(CLUSTER_PREFIX):
                    if len(curr_cluster) > 0:
                        epitopes_clusters_lst.append(curr_cluster)
                        curr_cluster = []
                else:
                    epitope_id = line.split(EPITOPE_ID_PREFIX)[1].split(EPITOPE_ID_SUFFIX)[0]
                    curr_cluster.append(epitope_id)

            # adding last cluster ser
            if len(curr_cluster) > 0:
                epitopes_clusters_lst.append(curr_cluster)

        return epitopes_clusters_lst
