# Optimal space decomposition based-product quantization for approximate nearest neighbor search

Abstract
Product quantization(PQ) is an effective nearest neighbor search (NNS) method for large-scale high-dimensional data. However, data quantization brings quantization error that may lower the retrieval accuracy. Many methods have been proposed. Among them, the method based on generating optimal PQ codes is very time and memory consuming. To address the problem, we theoretically prove that the more balanced the data volume in each subspace of product quantization is, the smaller the PQ quantization errors. Then an optimal space decomposition based-PQ (OSDPQ) algorithm is proposed. The algorithm solves the optimal space decomposition during product quantization by balancing the data volume in each subspace. Then, we propose the data retrieval method based on the quantization error (DRQE), which can effectively improve the retrieval accuracy of PQ-based NNS methods. Finally, the experimental results show that OSDPQ outperforms NNS methods based on PQ and neural network on 3 datasets. Comparing with the optimized product quantization (OPQ), the memory consumption of our method is reduced by 10%, and the speed of building indexing structure is increased by 10, 4 and 15 times under the close retrieval accuracy. Besides that, we verify the effectiveness of DRQE on PQ-based methods.  

## 2D graphic

![image](https://user-images.githubusercontent.com/38948350/141886048-b2a3ac5d-8321-4b9b-9632-9fff71174cc0.png)

## How to use
1. Preparation.
download dataSet: 链接：https://pan.baidu.com/s/1q66Xh-sDxJR5eVGUDib6ng 
                  提取码：8888 

3. Test
run OSDPQ.ipynb

3. Result

![Uploading image.png…]()

## Contacts
weilin  chen: weierLinC@163.com
shi  zhang: shi@fjnu.edu.cn
