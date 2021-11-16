import numpy as np
import os
# 产生我们数据的编码
def produce_Ints(idxMembers, sec_idxMembers, cluster_nums):
    # 用来存放我们的两个重要的数据
    Ints2Codebook_MemberCount = {}
    Ints2Codebook_Member = {}
    for key in idxMembers:
        for sec_key in sec_idxMembers:
            Ints = key*len(idxMembers) + sec_key
            if Ints not in Ints2Codebook_Member.keys():
                Ints2Codebook_Member[Ints] = []
            if Ints not in Ints2Codebook_MemberCount.keys():
                Ints2Codebook_MemberCount[Ints] = 0
            intersection_set = set(idxMembers[key]).intersection(set(sec_idxMembers[sec_key]))
            count = len(intersection_set)
            Ints2Codebook_MemberCount[Ints] = count
            Ints2Codebook_Member[Ints].append(list(intersection_set))
    if os.path.exists('./%d/output'%cluster_nums) == False:
        os.makedirs('./%d/output'%cluster_nums)
    np.savez_compressed('./%d/output/Ints2Codebook_MemberCount'%cluster_nums, Ints2Codebook_MemberCount = [Ints2Codebook_MemberCount])
    np.savez_compressed('./%d/output/Ints2Codebook_Member'%cluster_nums, Ints2Codebook_Member=[Ints2Codebook_Member])
    
    # arr = np.zeros((len(Ints2Codebook_MemberCount), 2))
    # for i, key in enumerate(Ints2Codebook_MemberCount):
    #     count = Ints2Codebook_MemberCount[key]
    #     arr[i][0] = key
    #     arr[i][1] = count
    # if os.path.exists('./%d/result'%cluster_nums) == False:
    #     os.makedirs('./%d/result'%cluster_nums)
    # np.savetxt('./%d/result/counter.txt'%cluster_nums, arr, delimiter=' ', fmt='%i')
    
    return Ints2Codebook_MemberCount, Ints2Codebook_Member