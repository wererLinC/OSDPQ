import math
import numpy as np
import math

def search_DRQE(dtable, N, M, P_errors, cluster_nums):
    Z = N*M
    M = len(P_errors)
    sort_index = np.argsort(P_errors)[::-1]
    Rs_arr = np.zeros((M, 1))
    i = 0
    for s, ps in enumerate(P_errors):
        theta_R = math.ceil(math.pow(Z, 1/(M-i)))
        if i == (M-1):
            Rs_arr[s] = Z
        # because M is small, so use -0.1*(ps)
        Rs = math.floor(np.exp(-0.1*(ps))*theta_R)
        Rs_arr[s] = Rs
        Z //= Rs
        i += 1
        
    query_clusters = []
    R1 = int(Rs_arr[0][0])
    R2 = int(Rs_arr[1][0])
    first_level_clusters = dtable[0][:R1]
    second_level_clusters = dtable[1][:R2]
    for f_cluster_id in first_level_clusters:
        for s_cluster_id in second_level_clusters:
            query_clusters.append(int(f_cluster_id*cluster_nums + s_cluster_id))
    return query_clusters



def search(dtable, N, M, cluster_nums):
    query_clusters = []
    first_level_clusters = dtable[0][:N]
    second_level_clusters = dtable[1][:M]
    for f_cluster_id in first_level_clusters:
        for s_cluster_id in second_level_clusters:
            query_clusters.append(int(f_cluster_id*cluster_nums + s_cluster_id))
    return query_clusters