from pyspark import SparkContext
import sys
import json
import time


def getCosineSim(set1, set2):
    if len(set1) != 0 and len(set2) != 0:
        p1 = set(set1)
        p2 = set(set2)
        num = len(p1 & p2)
        denum = pow(len(p1), 0.5)*pow(len(p2), 0.5)
        return float(num)/denum
    else:
        return 0


if __name__ == "__main__":
    st = time.time()
    sc = SparkContext("local[*]")
    output_file = sys.argv[3]
    test_file = sc.textFile(sys.argv[1]).map(lambda x: json.loads(x))\
        .map(lambda x: (x["user_id"], x["business_id"]))

    model_file = sc.textFile(sys.argv[2]).map(lambda x: json.loads(x))

    user_dict = model_file.filter(lambda x: x["cat"] == "Users")\
        .map(lambda x: (x["user_id"], x["user_index"])).collectAsMap()
    idx_user = {v1: k1 for k1, v1 in user_dict.items()}

    business = model_file.filter(lambda x: x["cat"] == "Business")\
        .map(lambda x: (x["business_id"], x["business_index"])).collectAsMap()
    idx_bus = {v1: k1 for k1, v1 in business.items()}

    user_profile = model_file.filter(lambda x: x["cat"] == "user_profile")\
        .map(lambda x: (x["user_index"], x["user_profile"])).collectAsMap()

    bus_profile = model_file.filter(lambda x: x["cat"] == "business_profile")\
        .map(lambda x: (x["business_index"], x["business_profile"])).collectAsMap()

    prediction = test_file.map(lambda x: (user_dict.get(x[0]), business.get(x[1])))\
        .filter(lambda x: x[0] is not None and x[1] is not None)\
        .map(lambda x: ((x[0], x[1]), getCosineSim(user_profile[x[0]], bus_profile[x[1]])))\
        .filter(lambda x: x[1] > 0.01)

    result = prediction.map(lambda x: {
        "user_id": idx_user[x[0][0]],
        "business_id": idx_bus[x[0][1]],
        "sim": x[1]
    }).collect()

    with open(output_file, "w") as f:
        for pred in result:
            f.write(json.dumps(pred) + "\n")
        f.close()

    print("Duration: ", time.time()-st)
