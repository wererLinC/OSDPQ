import numpy as np
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
    
    def get_codewords(self):
        return self.codewords
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
    
    
class OPQ(object):
    """Pure python implementation of Optimized Product Quantization (OPQ) [Ge14]_.
    OPQ is a simple extension of PQ.
    The best rotation matrix `R` is prepared using training vectors.
    Each input vector is rotated via `R`, then quantized into PQ-codes
    in the same manner as the original PQ.
    .. [Ge14] T. Ge et al., "Optimized Product Quantization", IEEE TPAMI 2014
    Args:
        M (int): The number of sub-spaces
        Ks (int): The number of codewords for each subspace (typically 256, so that each sub-vector is quantized
            into 256 bits = 1 byte = uint8)
        verbose (bool): Verbose flag
    Attributes:
        R (np.ndarray): Rotation matrix with the shape=(D, D) and dtype=np.float32
    """
    def __init__(self, M, Ks=256, verbose=True):
        self.pq = PQ(M, Ks, verbose)
        self.R = None
        self.err1 = 0
        self.err2 = 0

    def __eq__(self, other):
        if isinstance(other, OPQ):
            return self.pq == other.pq and np.array_equal(self.codewords, other.codewords)
        else:
            return False

    @property
    def M(self):
        """int: The number of sub-space"""
        return self.pq.M

    @property
    def Ks(self):
        """int: The number of codewords for each subspace"""
        return self.pq.Ks

    @property
    def verbose(self):
        """bool: Verbose flag"""
        return self.pq.verbose

    @property
    def code_dtype(self):
        """object: dtype of PQ-code. Either np.uint{8, 16, 32}"""
        return self.pq.code_dtype

    @property
    def codewords(self):
        """np.ndarray: shape=(M, Ks, Ds) with dtype=np.float32.
        codewords[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
        """
        return self.pq.codewords

    @property
    def Ds(self):
        """int: The dim of each sub-vector, i.e., Ds=D/M"""
        return self.pq.Ds

    # 这个其实只是多加了一个 rotation matrix ，训练的时候训练出来就是了
    def fit(self, vecs, pq_iter=20, rotation_iter=1, seed=123):
        """Given training vectors, this function alternatively trains
        (a) codewords and (b) a rotation matrix.
        The procedure of training codewords is same as :func:`PQ.fit`.
        The rotation matrix is computed so as to minimize the quantization error
        given codewords (Orthogonal Procrustes problem)
        This function is a translation from the original MATLAB implementation to that of python
        http://kaiminghe.com/cvpr13/index.html
        If you find the error message is messy, please turn off the verbose flag, then
        you can see the reduction of error for each iteration clearly
        Args:
            vecs: (np.ndarray): Training vectors with shape=(N, D) and dtype=np.float32.
            pq_iter (int): The number of iteration for k-means
            rotation_iter (int): The number of iteration for leraning rotation
            seed (int): The seed for random process
        Returns:
            object: self
        """
        assert vecs.ndim == 2
        _, D = vecs.shape
        self.R = np.eye(D)

        for i in range(rotation_iter):
            X = vecs @ self.R

            # (a) Train codewords
            pq_tmp = PQ(M=self.M, Ks=self.Ks, verbose=self.verbose)
            if i == rotation_iter - 1:
                # In the final loop, run the full training
                pq_tmp.fit(X, iter=pq_iter, seed=seed)
            else:
                # During the training for OPQ, just run one-pass (iter=1) PQ training
                pq_tmp.fit(X, iter=1, seed=seed)
            

            # (b) Update a rotation matrix R
            X_ = pq_tmp.decode(pq_tmp.encode(X))
            U, s, V = np.linalg.svd(vecs.T @ X_)
            codewords = pq_tmp.get_codewords()
            # 计算量化误差
            quantization_errors = np.linalg.norm(X - X_, 'fro')
            X_left = X[:, :D//2]
            X_right = X[:, D//2:]
            
            X__left = X_[:,:D//2]
            X__right = X_[:, D//2:]
            
            self.err1 = np.linalg.norm(X_left - X__left, 'fro')
            self.err2 = np.linalg.norm(X_right - X__right, 'fro')
            
            print("=========== PQ量化误差为:", quantization_errors, "===========")
#             mean = np.mean(X_, axis=0)
#             codewords__ = np.append(codewords[0], codewords[1], axis=1)
#             mean__ = mean.reshape((-1, D))
#             self.var_ = np.sum((codewords__ - mean__)**2)
#             # 打印簇间方差
#             print("==== 簇间方差:", self.var_, "比值：", (self.var_ / sss))
            if i == rotation_iter - 1:
                self.pq = pq_tmp
                break
            else:
                self.R = U @ V

        return self
    
    # 返回各个子空间的量化误差比例
    def get_errors_ratio(self):
        totalError = self.err1 + self.err2
        return [self.err1/(totalError), self.err2/(totalError)]

    def rotate(self, vecs):
        """Rotate input vector(s) by the rotation matrix.`
        Args:
            vecs (np.ndarray): Input vector(s) with dtype=np.float32.
                The shape can be a single vector (D, ) or several vectors (N, D)
        Returns:
            np.ndarray: Rotated vectors with the same shape and dtype to the input vecs.
        """
        assert vecs.ndim in [1, 2]

        if vecs.ndim == 2:
            return vecs @ self.R
        elif vecs.ndim == 1:
            return (vecs.reshape(1, -1) @ self.R).reshape(-1)

    def encode(self, vecs):
        """Rotate input vectors by :func:`OPQ.rotate`, then encode them via :func:`PQ.encode`.
        Args:
            vecs (np.ndarray): Input vectors with shape=(N, D) and dtype=np.float32.
        Returns:
            np.ndarray: PQ codes with shape=(N, M) and dtype=self.code_dtype
        """
        return self.pq.encode(self.rotate(vecs))

    def decode(self, codes):
        """Given PQ-codes, reconstruct original D-dimensional vectors via :func:`PQ.decode`,
        and applying an inverse-rotation.
        Args:
            codes (np.ndarray): PQ-cdoes with shape=(N, M) and dtype=self.code_dtype.
                Each row is a PQ-code
        Returns:
            np.ndarray: Reconstructed vectors with shape=(N, D) and dtype=np.float32
        """
        # Because R is a rotation matrix (R^t * R = I), R^-1 should be R^t
        return self.pq.decode(codes) @ self.R.T

    def dtable(self, query):
        """Compute a distance table for a query vector. The query is
        first rotated by :func:`OPQ.rotate`, then DistanceTable is computed by :func:`PQ.dtable`.
        Args:
            query (np.ndarray): Input vector with shape=(D, ) and dtype=np.float32
        Returns:
            nanopq.DistanceTable:
                Distance table. which contains
                dtable with shape=(M, Ks) and dtype=np.float32
        """
        return self.pq.dtable(query)