import numpy as np
from time import time
Tstart_time = time()

def Searching_Top100(Ints_clustMem, Top100, idxCount, InsIdx2):
    '''
        InsIdx2: donate as  clusters of all query data type:[]
    '''
          
    recall = 0
    # 拿出列表的元素了
    total_counter = 0
    for i in InsIdx2: 
        if i not in Ints_clustMem.keys():
            continue
        ints_member = Ints_clustMem[i][0]
        total_counter = total_counter + idxCount[i]
        recall = recall + len(np.intersect1d(Top100, ints_member))
    return recall, total_counter