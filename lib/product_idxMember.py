import numpy as np
def produce_idxCount_idxMembers(codes):
    '''
        输入的是各个阶段的码书
    '''
    idxMembers = {}
    for idx, cluster_id in enumerate(codes):
        if cluster_id not in idxMembers.keys():
            idxMembers[cluster_id] = []
        # 簇心ID相等的个数进行++
        idxMembers[cluster_id].append(idx)
    return idxMembers