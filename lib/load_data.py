import numpy as np

def load_data(filename="gist"):
    '''
        using FMA:filename = "fma"
        using GIST1M:filename = "gist"
        using MNIST:filename = "mnist"
    '''
    print("############# loading %s dataset......#############"%filename)
    base_data = np.loadtxt('./dataSet/%s_pca_data.txt'%filename)
    query_data = np.loadtxt('./dataSet/%s_query_data.txt'%filename)
    Top100 = np.loadtxt('./dataSet/%s_Top100.txt'%filename)
    return base_data, query_data, Top100