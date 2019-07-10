from collections import defaultdict
import pandas as pd
import numpy as np
import time
import spacy

nlp = spacy.load('en_core_web_sm')


df = pd.read_excel('./询盘内容关联可疑样本.xlsx')
sample_1 = df.iloc[1, 2]
sample_2 = df.iloc[1, 5]
sample_3 = df.iloc[1:3, 8].values.tolist()
sample_4 = df.iloc[1, 11]
sample_5 = df.iloc[1:3, 14].values.tolist()
sample_6 = df.iloc[1:4, 17].values.tolist()
sample_7 = df.iloc[1:4, 20].values.tolist()


if __name__ == "__main__":
    from shingling import MinHash
    from lsh import LSH

    samples = []
    hash_objs = []
    for sample in [sample_1, sample_2, sample_3, sample_4, sample_5, sample_6, sample_7]:
        if type(sample) is str:
            samples.append(sample)
            hash_objs.append(MinHash(sample, 3, 1000))
        elif type(sample) is list:
            for s in sample:
                samples.append(s)
                hash_objs.append(MinHash(s, 3, 1000))
    
    signature_matrix = MinHash.build_signature_matrix(hash_objs)
    # print(signature_matrix.shape)

    tick = time.time()
    for sample in samples:
        min_hash = MinHash(sample, 3, 1000)
        MinHash.find_near_duplicates(np.array(min_hash.min_hash), signature_matrix, 0.6)
    # for i in range(signature_matrix.shape[0]):
    #     print(MinHash.find_near_duplicates(signature_matrix[i], signature_matrix, 0.6))
    tock = time.time()
    print(tock - tick)
    
    lsh = LSH(signature_matrix, 20, 1)
    shared_counts = defaultdict(lambda: defaultdict(int))

    tick = time.time()
    for value, bucket in lsh.hash_table.items():
        if len(bucket) > 1:
            for sample_id_a, _ in bucket:
                for sample_id_b, __ in bucket:
                    if sample_id_a != sample_id_b:
                        shared_counts[sample_id_a][sample_id_b] += 1
    tock = time.time()
    print(tock - tick)
    # for sampled_id, shared_dict in sorted(shared_counts.items()):
    #     print('****************')
    #     print(sampled_id)
    #     print('----------------')
    #     for shared_sample_id, counts in sorted(shared_dict.items()):
    #         if counts > 1:
    #             print(shared_sample_id, counts)