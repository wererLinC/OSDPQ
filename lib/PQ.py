import numpy as np
from scipy.cluster.vq import vq, kmeans2


class PQ(object):
    """Pure python implementation of Product Quantization (PQ) [Jegou11]_.
    For the indexing phase of database vectors,
    a `D`-dim input vector is divided into `M` `D`/`M`-dim sub-vectors.
    Each sub-vector is quantized into a small integer via `Ks` codewords.
    For the querying phase, given a new `D`-dim query vector, the distance beween the query
    and the database PQ-codes are efficiently approximated via Asymmetric Distance.
    All vectors must be np.ndarray with np.float32
    .. [Jegou11] H. Jegou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011
    Args:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
            (typically 256, so that each sub-vector is quantized
            into 256 bits = 1 byte = uint8)
        verbose (bool): Verbose flag
    Attributes:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
        verbose (bool): Verbose flag
        code_dtype (object): dtype of PQ-code. Either np.uint{8, 16, 32}
        codewords (np.ndarray): shape=(M, Ks, Ds) with dtype=np.float32. 原来是3维的这样的，是不是呢
            codewords[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
        Ds (int): The dim of each sub-vector, i.e., Ds=D/M
    """
    def __init__(self, M=8, Ks=256, verbose=True):
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.verbose = M, Ks, verbose
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        self.codewords = None
        self.Ds = None

        if verbose:
            print("M: {}, Ks: {}, code_dtype: {}".format(M, Ks, self.code_dtype))

    def __eq__(self, other):
        if isinstance(other, PQ):
            return (self.M, self.Ks, self.verbose, self.code_dtype, self.Ds) == \
                   (other.M, other.Ks, other.verbose, other.code_dtype, other.Ds) and \
                   np.array_equal(self.codewords, other.codewords)
        else:
            return False

    def fit(self, vecs, iter=20, seed=123):
        """Given training vectors, run k-means for each sub-space and create
        codewords for each sub-space.
        This function should be run once first of all.
        Args:
            vecs (np.ndarray): Training vectors with shape=(N, D) and dtype=np.float32.
            iter (int): The number of iteration for k-means
            seed (int): The seed for random process
        Returns:
            object: self
        """
        # 居然是用训练集来做fit的，所以训练出来的码书，也是跟 train_data 有关而已
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        assert D % self.M == 0, "input dimension must be dividable by M"
        self.Ds = int(D / self.M)

        np.random.seed(seed)
        if self.verbose:
            print("iter: {}, seed: {}".format(iter, seed))

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        self.codewords = np.zeros((self.M, self.Ks, self.Ds), dtype=np.float32)
        for m in range(self.M):
            if self.verbose:
                print("Training the subspace: {} / {}".format(m, self.M))
            vecs_sub = vecs[:, m * self.Ds : (m+1) * self.Ds]
            self.codewords[m], _ = kmeans2(vecs_sub, self.Ks, iter=iter, minit='points')

        return self

    def encode(self, vecs):
        """Encode input vectors into PQ-codes.
        Args:
            vecs (np.ndarray): Input vectors with shape=(N, D) and dtype=np.float32.
        Returns:
            np.ndarray: PQ codes with shape=(N, M) and dtype=self.code_dtype
        """
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert D == self.Ds * self.M, "input dimension must be Ds * M"

        # codes[n][m] : code of n-th vec, m-th subspace
        codes = np.empty((N, self.M), dtype=self.code_dtype) 
        for m in range(self.M):
            if self.verbose:
                print("Encoding the subspace: {} / {}".format(m, self.M))
            vecs_sub = vecs[:, m * self.Ds : (m+1) * self.Ds]
            codes[:, m], _ = vq(vecs_sub, self.codewords[m]) # 得到我们的编码

        return codes

    def decode(self, codes):
        """Given PQ-codes, reconstruct original D-dimensional vectors
        approximately by fetching the codewords.
        Args:
            codes (np.ndarray): PQ-cdoes with shape=(N, M) and dtype=self.code_dtype.
                Each row is a PQ-code
        Returns:
            np.ndarray: Reconstructed vectors with shape=(N, D) and dtype=np.float32
        """
        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.M

        vecs = np.empty((N, self.Ds * self.M))
        for m in range(self.M):
            vecs[:, m * self.Ds : (m+1) * self.Ds] = self.codewords[m][codes[:, m], :]

        return vecs
    
    def dtable(self, query):
        '''
            可能我需要进行返回并不是是最近的质点，而是要多个才行
        '''
        # self.codewords[m] 是256*（128/8）的一个数组
        dtable = np.empty((self.M, self.Ks))
        result_cluster = np.empty((self.M, self.Ks))
        for m in range(self.M):
            query_sub = query[m * self.Ds : (m+1) * self.Ds] # 对我们的query数据进行分块
            # 得到每个查询到每个质心的距离
            dtable[m, :] = np.linalg.norm(self.codewords[m] - query_sub, axis=1) ** 2
            result_cluster[m] = np.argsort(dtable[m])  # 得到最近的cluster
        return result_cluster