import numpy as np

def OSDPQ(base_data, query_data, times=1):
    '''
    注意：
     如果是GIST数据集，那么times设置为1000，因为其方差值很多都是小于0
    '''
    print("############# Doing OSDPQ Space Decomposition ##################")
    explained_variance = np.var(base_data, axis=0)
    explained_variance = np.sort(explained_variance)
    # 因为GIST1M方差太小，所以所有维度乘上100倍
    explained_variance *= times
    lens = len(explained_variance)-1
    result1 = explained_variance[0] * explained_variance[lens]
    result2 = explained_variance[1] * explained_variance[lens-1]
    list1 = [0, 1]
    for i in range(2, (len(explained_variance)//2)):
        if(result1 < result2):
            result1 *= explained_variance[i]*explained_variance[lens-i]
            list1.append(0);
        else:
            result2 *= explained_variance[i]*explained_variance[lens-i]
            list1.append(1);
    list2 = list(list1 + list1[::-1])
    index = np.array(list2)
    
    base_data_left = base_data[:, np.where(index == 1)]
    base_data_right = base_data[:, np.where(index == 0)]

    query_data_left = query_data[:, np.where(index == 1)]
    query_data_right = query_data[:, np.where(index == 0)]

    base_data_left = base_data_left.reshape(base_data_left.shape[0], base_data_left.shape[2])
    query_data_left = query_data_left.reshape(query_data_left.shape[0], query_data_left.shape[2])

    base_data_right = base_data_right.reshape(base_data_right.shape[0], base_data_right.shape[2])
    query_data_right = query_data_right.reshape(query_data_right.shape[0], query_data_right.shape[2])

    osd_base_data = np.append(base_data_left, base_data_right, axis=1)
    osd_query_data = np.append(query_data_left, query_data_right, axis=1)
    print("############# OSDPQ Space Decomposition End ##################")
    return osd_base_data, osd_query_data