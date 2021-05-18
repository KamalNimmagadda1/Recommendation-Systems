from pyspark import SparkContext
import sys
import json
import random
import time
from itertools import combinations

HASH_FUNC = 100
a = random.sample(range(0, sys.maxsize-1), HASH_FUNC)
b = random.sample(range(1, sys.maxsize-1), HASH_FUNC)


def genHashFunc(numUsers):
    hashList = []
    def hash1(i, j):
        def hashed(param_x):
            p = ((i * param_x + j) % 99999989) % numUsers
            return p
        return hashed

    for i, j in zip(a, b):
        hashList.append(hash1(i, j))

    return hashList


def simplifyCol(l1, l2):
    lst = [min(x, y) for x, y in zip(l1, l2)]
    return lst


def getBands(lsts, bandNum):
    band = []
    len1 = int(len(lsts) / bandNum)
    for idx, i in enumerate(range(0, len(lsts), len1)):
        band.append((idx, hash(tuple(lsts[i:i + len1]))))
    return band


def computeJacc(cands, data, log1):
    res = []
    temp_set = set()
    for pair in cands:
        if pair not in temp_set:
            temp_set.add(pair)
            sim = float(float(len(set(data.get(pair[0])) & set(data.get(pair[1])))) / len(set(data.get(pair[0])) | set(data.get(pair[1]))))
            if sim >= 0.05:
                res.append({"b1": log1[pair[0]],
                            "b2": log1[pair[1]],
                            "sim": sim})
    return res


if __name__ == "__main__":
    st = time.time()
    sc = SparkContext("local[*]")
    jacc_sim_threshold = 0.05
    output_file = sys.argv[2]

    input_file = sc.textFile(sys.argv[1]).map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], x["user_id"])).persist()
    users = input_file.map(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex().map(
        lambda x: (x[0], x[1]))
    user_dict = users.collectAsMap()
    business = input_file.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex().map(
        lambda x: (x[0], x[1])).collectAsMap()
    rev_bus = {val: key for key, val in business.items()}
    hashFunctions = genHashFunc(len(user_dict))
    hashed_users = users.map(lambda x: (user_dict[x[0]], [hashed(x[1]) for hashed in hashFunctions]))
    dataRDD = input_file.map(lambda x: (user_dict[x[1]], business[x[0]])).groupByKey().map(
        lambda x: (x[0], list(set(x[1]))))
    bus_to_user = input_file.map(lambda x: (business[x[0]], user_dict[x[1]])).groupByKey().map(
        lambda x: (x[0], list(set(x[1])))).collectAsMap()
    signaturesRDD = dataRDD.leftOuterJoin(hashed_users).map(lambda x: x[1]).flatMap(
        lambda x: [(i, x[1]) for i in x[0]]).reduceByKey(simplifyCol).coalesce(2)
    bands = signaturesRDD.flatMap(lambda x: [(tuple(ite), x[0]) for ite in getBands(x[1], 100)]).groupByKey().map(
        lambda x: list(x[1])).filter(lambda bnd: len(bnd) > 1)
    cand_pairs = bands.flatMap(lambda x: [p for p in combinations(x, 2)]).collect()
    result = computeJacc(set(cand_pairs), bus_to_user, rev_bus)

    with open(output_file, "w") as f:
        for i in result:
            f.write(json.dumps(i) + "\n")
        f.close()
    # Totaltime = time.time() - st
    # print(f"Duration: {Totaltime}")