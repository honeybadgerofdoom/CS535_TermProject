import pymongo


def getDB():
    mongo = pymongo.MongoClient(f'mongodb_atlas://cs535_tp:superlemur55@mongo_atlas:41718/')
    db = mongo['termProj']
    return db


def main():
    db = getDB()
    algae_code(db)


def algae_code(db):
    map = {
        None: "no",
        "None": "no",
        "Mild": "no",
        "Moderate": "yes",
        "Serious": "yes",
        "Extreme": "yes"
    }
    for key in map:
        value = map[key]
        db["wq_alg"].update_many(
            { "$and": [ { "category": "algae" }, { "unit": "code" }, { "value": key } ] },
            { "$set": { "value": value, "unit": "yes/no" } }
        )



if __name__ == "__main__":
    main()

