from pyspark import SparkContext
import sys
import json
import string
import math
import time


def getToken(iterator, stopwords):
    text_data = []
    for i in iterator:
        text_data.append(i)
    temp = []
    for text in text_data:
        p = text.translate(str.maketrans("", "", string.digits + string.punctuation)).split()
        t = []
        for i in p:
            if i not in stopwords and ['']:
                t.append(i)
        temp.append(t)
    return temp


def getWordCount(iterator):
    text_data = []
    for i in iterator:
        text_data.append(i)
    text_dict = dict()
    max_words = 0
    for text in text_data:
        max_words += len(text)
        for word in text:
            if word not in text_dict.keys():
                text_dict[word] = 0
            text_dict[word] += 1
    max_app = max([v for v in text_dict.values()])
    text_dict = dict(filter(lambda val: float(float(val[1])/max_words) > 0.0001, text_dict.items()))
    res = []
    for k, v in text_dict.items():
        res.append([k, v, max_app])
    return sorted(res, key=lambda x: x[1], reverse=True)


def getProf(iterator, uid_dict, bid_dict, bid_profile):
    data_chunk = []
    for id1 in iterator:
        uidx = uid_dict[id1[0]]
        data_chunk.append((uidx, id1[1]))
    res = dict()
    for data in data_chunk:
        uprof_list = []
        for idx in data[1]:
            bidx = bid_dict[idx]
            uprof = bid_profile[bidx]
            uprof_list.extend(uprof)
        res[data[0]] = sorted(set(uprof_list))
    return res


if __name__ == "__main__":
    st = time.time()
    sc = SparkContext("local[*]")
    model_file = sys.argv[2]
    stopwords = sc.textFile(sys.argv[3]).flatMap(lambda x: x.split("\n")).collect()
    # print(stopwords[0:5])
    model = []

    input_file = sc.textFile(sys.argv[1]).map(lambda x: json.loads(x))

    users = input_file.map(lambda x: x["user_id"]).distinct().sortBy(lambda x: x).zipWithIndex().map(
        lambda x: (x[0], x[1]))
    user_dict = users.collectAsMap()

    if type(user_dict) == dict:
        for key, val1 in user_dict.items():
            model.append({
                "cat": "Users",
                "user_id": key,
                "user_index": val1
            })

    business = input_file.map(lambda x: x["business_id"]).distinct().sortBy(lambda x: x).zipWithIndex().map(
        lambda x: (x[0], x[1])).collectAsMap()

    if type(business) == dict:
        for k1, v1 in business.items():
            model.append({
                "cat": "Business",
                "business_id": k1,
                "business_index": v1
            })

    tfRDD = input_file.map(lambda line: (business[line["business_id"]], str(line["text"]).lower()))\
        .groupByKey().map(lambda x: (x[0], getToken(x[1], stopwords)))\
        .map(lambda x: (x[0], getWordCount(x[1])))\
        .flatMap(lambda x: [((x[0], v[0]), float(v[1])/v[2]) for v in x[1]]).persist()

    idfRDD = tfRDD.map(lambda x: (x[0][1], x[0][0])).groupByKey()\
        .map(lambda x: (x[0], list(set(x[1]))))\
        .flatMap(lambda x: [((i, x[0]), math.log2(float(len(business)/len(x[1])))) for i in x[1]]).persist()

    tf_idf = tfRDD.leftOuterJoin(idfRDD).map(lambda x: (x[0][0], (x[0][1], x[1][0] * x[1][1])))\
        .groupByKey().map(lambda x: (x[0], sorted(list(x[1]), key=lambda i1: i1[1], reverse=True)[:200]))\
        .map(lambda p: (p[0], [word[0] for word in p[1]]))
    # print(tf_idf.first())

    words = tf_idf.flatMap(lambda x: [wrd for wrd in x[1]]).zipWithIndex().map(lambda x: (x[0], x[1])).collectAsMap()

    business_profile = tf_idf.map(lambda x: {x[0]: [words[word] for word in x[1]]}).collect()
    bpro_data = dict()
    for profile in business_profile:
        bpro_data[list(profile.keys())[0]] = list(profile.values())[0]

    if type(business_profile) == list:
        for profile in business_profile:
            for bus, w in profile.items():
                model.append({
                    "cat": "business_profile",
                    "business_index": bus,
                    "business_profile": sorted(w)
                })

    user_prof = input_file.map(lambda x: (x["user_id"], x["business_id"])).groupByKey()\
        .map(lambda x: (x[0], list(set(x[1])))).collect()
    user_profile = getProf(user_prof, user_dict, business, bpro_data)
    print("here", type(user_profile), len(user_profile))

    if type(user_profile) == dict:
        for user, prof in user_profile.items():
            model.append({
                "cat": "user_profile",
                "user_index": user,
                "user_profile": prof
            })

    with open(model_file, "w") as f:
        for i in model:
            f.write(json.dumps(i) + "\n")
        f.close()
    print("Duration:", time.time()-st)
