from pyspark import SparkContext
import json
import sys
import time
import random
from itertools import combinations

HASH_FUNC = 40
a = random.sample(range(0, sys.maxsize - 1), HASH_FUNC)
b = random.sample(range(1, sys.maxsize - 1), HASH_FUNC)


def genHash(numUsers):
    hashList = []

    def hash1(i, j):
        def hashed(param_x):
            p = ((i * param_x + j) % 99999989) % numUsers
            return p

        return hashed

    for i, j in zip(a, b):
        hashList.append(hash1(i, j))

    return hashList


def unify(lst):
    unidict = dict()
    for i in lst:
        unidict[i[0]] = i[1]
    return unidict


def simplifyCol(l1, l2):
    lst = [min(x, y) for x, y in zip(l1, l2)]
    return lst


def getBands(lsts, bandNum):
    band = []
    len1 = int(len(lsts) / bandNum)
    for idx, i in enumerate(range(0, len(lsts), len1)):
        band.append((idx, hash(tuple(lsts[i:i + len1]))))
    return band


def applyHash(func, idx):
    lst = list(map(lambda x: x(idx), func))
    return lst


def computeCosineSim(idt1, idt2):
    corated = list(set(idt1.keys()) & set(idt2.keys()))
    l1 = [idt1[i1] for i1 in corated]
    l2 = [idt2[i1] for i1 in corated]

    mean1 = sum(l1) / len(l1)
    mean2 = sum(l2) / len(l2)

    numerator = 0
    for i in zip(l1, l2):
        numerator += (i[0] - mean1) * (i[1] - mean2)
    if numerator == 0:
        return 0

    denum1 = 0
    for i1 in l1:
        denum1 += pow((i1 - mean1), 2)
    denum1 = pow(denum1, 0.5)
    denum2 = 0
    for i1 in l2:
        denum2 += pow((i1 - mean2), 2)
    denum2 = pow(denum2, 0.5)

    denominator = denum1 * denum2
    if denominator == 0:
        return 0

    return numerator / denominator


def computeJaccard(d1, d2):
    if d1 is not None and d2 is not None:
        u1 = set(d1.keys())
        u2 = set(d2.keys())
        if len(u1 & u2) >= 3:
            if float(float(len(u1 & u2)) / len(u1 | u2)) >= 0.01:
                return True
    return False


if __name__ == "__main__":
    st = time.time()
    sc = SparkContext("local[*]")

    input_file = sc.textFile(sys.argv[1]).map(lambda x: json.loads(x)) \
        .map(lambda x: (x["user_id"], x["business_id"], x["stars"]))
    output_file = sys.argv[2]
    model_type = str(sys.argv[3])

    user_dict = input_file.map(lambda x: x[0]).distinct() \
        .sortBy(lambda x: x).zipWithIndex() \
        .map(lambda x: (x[0], x[1])).collectAsMap()
    idx_user = {val: key1 for key1, val in user_dict.items()}

    business = input_file.map(lambda x: x[1]).distinct() \
        .sortBy(lambda x: x).zipWithIndex() \
        .map(lambda x: (x[0], x[1])).collectAsMap()
    idx_bus = {v1: k1 for k1, v1 in business.items()}

    bus_rat = input_file.map(lambda x: (business[x[1]], (user_dict[x[0]], x[2]))) \
        .groupByKey().map(lambda x: (x[0], list(x[1]))) \
        .filter(lambda x: len(x[1]) >= 3).persist()

    if model_type == "item_based":
        bus_rat_dict = bus_rat.map(lambda x: (x[0], [(v[0], v[1]) for v in x[1]])).mapValues(lambda x: unify(x))
        # print(bus_rat_dict.first())

        bus_cand = bus_rat_dict.map(lambda x: x[0]).coalesce(2)
        bus_user_dict = bus_rat_dict.collectAsMap()

        candidates = bus_cand.cartesian(bus_cand).filter(lambda x: x[0] < x[1]) \
            .filter(lambda x: len(set(bus_user_dict[x[0]].keys()) & set(bus_user_dict[x[1]].keys())) >= 3) \
            .map(lambda x: ((x[0], x[1]), computeCosineSim(bus_user_dict[x[0]], bus_user_dict[x[1]]))) \
            .filter(lambda x: x[1] > 0)

        cand_pairs = candidates.map(lambda x: {
            "b1": idx_bus[x[0][0]],
            "b2": idx_bus[x[0][1]],
            "sim": x[1]
        }).collect()

        with open(output_file, "w") as f:
            for i in cand_pairs:
                f.write(json.dumps(i) + "\n")
            f.close()

    if model_type == "user_based":
        hashes = genHash(len(business))
        # print("hashes here")

        hashed_user = bus_rat.flatMap(lambda x: [(v[0], applyHash(hashes, x[0])) for v in x[1]])\
            .reduceByKey(simplifyCol).flatMap(lambda x: [(tuple(ite), x[0]) for ite in getBands(x[1], HASH_FUNC)])\
            .groupByKey().map(lambda x: sorted(set(x[1]))).filter(lambda bnd: len(bnd) > 1)

        usr_pair = hashed_user.flatMap(lambda x: [p for p in combinations(x, 2)]).distinct()
        # print("user pairs here", usr_pair.first())

        user_bus_dict = bus_rat.flatMap(lambda x: [(v[0], (x[0], v[1])) for v in x[1]]).groupByKey()\
            .map(lambda x: (x[0], list(set(x[1])))).filter(lambda x: len(x[1]) >= 3)\
            .map(lambda x: (x[0], [(v[0], v[1]) for v in x[1]])).mapValues(lambda x: unify(x))\
            .map(lambda x: (x[0], x[1])).collectAsMap()
        # print("ubd", len(user_bus_dict))

        user_cand = usr_pair.filter(lambda x: computeJaccard(user_bus_dict.get(x[0]), user_bus_dict.get(x[1])))\
            .map(lambda x: ((x[0], x[1]), computeCosineSim(user_bus_dict[x[0]], user_bus_dict[x[1]])))\
            .filter(lambda x: x[1] > 0)

        cand_pairs = user_cand.map(lambda x: {
            "u1": idx_user[x[0][0]],
            "u2": idx_user[x[0][1]],
            "sim": x[1]
        }).collect()

        with open(output_file, "w") as f:
            for i in cand_pairs:
                f.write(json.dumps(i) + "\n")
            f.close()

    print("Duration: ", time.time() - st)
