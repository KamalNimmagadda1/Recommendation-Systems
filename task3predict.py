from pyspark import SparkContext
import json
import sys
import time

N = 4
AVG_STARS = 3.725671


def getPrediction(data, avg_dict, model, idx_dict, type):
    if type == "item_based":
        bus_id = idx_dict.get(data[0])
        bus_stars = list(data[1])
        res = []
        for star in bus_stars:
            if data[0] < star[0]:
                p = tuple((data[0], star[0]))
            else:
                p = tuple((star[0], data[0]))
            res.append(tuple((star[1], model.get(p, 0))))

        star_list = sorted(res, key=lambda x: x[1], reverse=True)[:N]
        numerator = 0
        for i in star_list:
            numerator += i[0] * i[1]
        if numerator == 0:
            return tuple((data[0], avg_dict.get(bus_id, AVG_STARS)))
        denominator = 0
        for i in star_list:
            denominator += abs(i[1])
        if denominator == 0:
            return tuple((data[0], avg_dict.get(bus_id, AVG_STARS)))
        return tuple((data[0], numerator / denominator))

    else:
        usr_id = idx_dict.get(data[0])
        usr_stars = list(data[1])
        res = []
        for star in usr_stars:
            if data[0] < star[0]:
                p = tuple((data[0], star[0]))
            else:
                p = tuple((star[0], data[0]))
            o_star = idx_dict.get(star[0], AVG_STARS)
            avg_val = avg_dict.get(o_star)
            res.append(tuple((star[1], avg_val, model.get(p, 0))))

        numerator = 0
        for x in res:
            numerator += (x[0] - x[1]) * x[2]
        if numerator == 0:
            return tuple((data[0], avg_dict.get(usr_id, AVG_STARS)))
        denominator = 0
        for x in res:
            denominator += abs(x[2])
        if denominator == 0:
            return tuple((data[0], avg_dict.get(usr_id, AVG_STARS)))
        return tuple((data[0], avg_dict.get(usr_id, AVG_STARS) + (numerator / denominator)))


if __name__ == "__main__":
    st = time.time()
    sc = SparkContext("local[*]")
    train_file = sc.textFile(sys.argv[1]).map(lambda x: json.loads(x))\
        .map(lambda x: (x["user_id"], x["business_id"], x["stars"]))
    model_type = sys.argv[5]
    test_file = sc.textFile(sys.argv[2]).map(lambda x: json.loads(x))
    output_file = sys.argv[4]

    user_dict = train_file.map(lambda x: x[0]).distinct() \
        .sortBy(lambda x: x).zipWithIndex() \
        .map(lambda x: (x[0], x[1])).collectAsMap()
    idx_user = {val: key1 for key1, val in user_dict.items()}

    business = train_file.map(lambda x: x[1]).distinct() \
        .sortBy(lambda x: x).zipWithIndex() \
        .map(lambda x: (x[0], x[1])).collectAsMap()
    idx_bus = {v1: k1 for k1, v1 in business.items()}

    if model_type == "item_based":
        model_file = sc.textFile(sys.argv[3]).map(lambda x: json.loads(x))\
            .map(lambda x: ((business[x["b1"]], business[x["b2"]]), x["sim"])).collectAsMap()

        training = train_file.map(lambda x: (user_dict[x[0]], (business[x[1]], x[2]))).groupByKey() \
            .mapValues(lambda x: [(v[0], v[1]) for v in list(set(x))])

        test_bus = test_file.map(lambda x: (user_dict.get(x["user_id"]), business.get(x["business_id"])))\
            .filter(lambda x: x[0] is not None and x[1] is not None)

        bus_avg = sc.textFile("/Users/kamalnimmagadda/IdeaProjects/ass3/data/business_avg.json")\
            .map(lambda x: json.loads(x)).map(lambda x: dict(x)).flatMap(lambda x: x.items()).collectAsMap()

        pred_pair = test_bus.leftOuterJoin(training)\
            .mapValues(lambda x: getPrediction(tuple(x), bus_avg, model_file, idx_bus, model_type))

        prediction = pred_pair.map(lambda x: {
            "user_id": idx_user[x[0]],
            "business_id": idx_bus[x[1][0]],
            "stars": x[1][1]
        }).collect()

        with open(output_file, "w") as f:
            for i in prediction:
                f.write(json.dumps(i) + "\n")
            f.close()

    if model_type == "user_based":
        model_file = sc.textFile(sys.argv[3]).map(lambda x: json.loads(x))\
            .map(lambda x: ((user_dict[x["u1"]], user_dict[x["u2"]]), x["sim"])).collectAsMap()

        training = train_file.map(lambda x: (business[x[1]], (user_dict[x[0]], x[2]))).groupByKey()\
            .mapValues(lambda x: [(v[0], v[1]) for v in list(set(x))])

        usr_avg = sc.textFile("/Users/kamalnimmagadda/IdeaProjects/ass3/data/user_avg.json")\
            .map(lambda x: json.loads(x)).map(lambda x: dict(x)).flatMap(lambda x: x.items()).collectAsMap()

        test_usr = test_file.map(lambda x: (business.get(x["business_id"]), user_dict.get(x["user_id"]))) \
            .filter(lambda x: x[0] is not None and x[1] is not None)

        pred_pair = test_usr.leftOuterJoin(training)\
            .mapValues(lambda x: getPrediction(tuple(x), usr_avg, model_file, idx_user, model_type))

        prediction = pred_pair.map(lambda x: {
            "user_id": idx_user[x[1][0]],
            "business_id": idx_bus[x[0]],
            "stars": x[1][1]
        }).collect()

        with open(output_file, "w") as f:
            for i in prediction:
                f.write(json.dumps(i) + "\n")
            f.close()