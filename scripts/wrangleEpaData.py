import pymongo
import json


measurements = ["nitrate", "phosphorus"]
nitr = ".*nitr.*"
phosph = ".*phosph.*"
alga = ".*alga.*"
flow = ".*flow.*"


def main():
    orderBySite()


def orderBySite():
    with open("savedData/measurementData.json", "r") as f:
        data = json.loads(f.read())
    allData = {}
    for datum in data:
        id = datum["monitorId"]
        del datum["regexMatched"]
        if id in allData:
            allData[id].append(datum)
        else:
            allData[id] = [datum]
    for key in allData:
        value = allData[key]
        writeToFile(f"siteData/{key}", value)


def buildNewCollection():
    db = getDB()
    with open("intersectedSites.json", "r") as f:
        intersectedSites = json.loads(f.read())
    matchSites = getMatches(intersectedSites, "MonitoringLocationIdentifier")
    regex = f"{nitr}|{phosph}|{alga}|{flow}"
    documents = list(db["aqua_measurements"].aggregate([
        matchSites,
        { "$addFields": { "regexMatched": { "$regexMatch": { "input": "$measurement_name", "regex": regex, "options": "i" }  } } },
        { "$match": { "$or": [ { "regexMatched": True }, { "measurement_name": "temperature, water" }, { "measurement_name": "ph" } ] } }
    ]))
    for document in documents:
        formatDocument(document)
        del document["regexMatched"]
    print(f"Found {len(documents)} results")
    writeToFile("measurementData", documents)


def getIntersectedSites():
    sitesWithNitr = getSitesByRegex(nitr)
    sitesWithPhosph = getSitesByRegex(phosph)
    sitesWithAlga = getSitesByRegex(alga)

    intersection = sitesWithNitr.intersection(sitesWithPhosph, sitesWithAlga)
    writeToFile("intersectedSites", list(intersection))


def getSitesByRegex(regex):
    results = []

    db = getDB()
    documents = list(db["aqua_measurements"].aggregate([
        { "$group": { "_id": "$measurement_name" } },
        { "$project": { "_id": 0, "measurement_name": "$_id" } },
        { "$addFields": { "include": { "$regexMatch": { "input": "$measurement_name", "regex": regex, "options": "i" }  } } },
        { "$match": { "include": True } },
        { "$project": { "include": 0 } },
    ]))

    for document in documents:
        measurement = document["measurement_name"]
        sites = list(db["aqua_measurements"].aggregate([
            { "$match": { "measurement_name": measurement } },
            { "$group": { "_id": "$MonitoringLocationIdentifier" } },
        ]))
        results.extend(sites)
    return set([result["_id"] for result in results])


def aggregateAlgaeSites():
    existing = []
    with open("nitrate_phosphorus.json", "r") as f:
        existing = json.loads(f.read())
    existingSitesSet = set(entry["site"] for entry in existing)

    db = getDB()
    input = []
    with open("algaeSitesAndCounts.json", "r") as f:
        input = json.loads(f.read())

    allSites = []

    for site in input:
        sites = [entry["MonitoringLocationIdentifier"] for entry in site["sites"]]
        filteredSites = []
        for entry in sites:
            if entry in existingSitesSet:
                filteredSites.append(entry)
        if len(filteredSites) < 1:
            continue

        allSites.extend(filteredSites)
    print(allSites)

    matches = getMatches(allSites, "MonitoringLocationIdentifier")
    measurements = list(db["aqua_measurements"].aggregate([
        matches
    ]))
    for measurement in measurements:
        formatDocument(measurement)
    writeToFile("siteMeasurements", measurements)

    '''
        Next, get all .*agla.* data from these 6 sites! Maybe need to add synthetic data here.
        ['USGS-06313400', 'USGS-06607500', 'USGS-05418600', 'USGS-02300200', 'USGS-08181500', 'USGS-06313400']
    '''


def findAlgaeDataSites():
    db = getDB()

    documents = list(db["aqua_measurements"].aggregate([
        { "$group": { "_id": "$measurement_name" } },
        { "$project": { "_id": 0, "measurement_name": "$_id" } },
        { "$addFields": { "include": { "$regexMatch": { "input": "$measurement_name", "regex": ".*alga.*", "options": "i" }  } } },
        { "$match": { "include": True } },
        { "$project": { "include": 0 } },
    ]))

    results = []
    for document in documents:
        measurement = document["measurement_name"]
        sites = list(db["aqua_measurements"].aggregate([
            { "$match": { "measurement_name": measurement } },
            { "$group": { "_id": "$MonitoringLocationIdentifier", "count": { "$sum": 1 } } },
            { "$project": { "_id": 0, "MonitoringLocationIdentifier": "$_id", "count": 1 } },
            { "$sort": { "count": -1 } }
        ]))
        results.append({
            "measurement_name": measurement,
            "sites": sites
        })
    writeToFile("algaeSitesAndCounts", results)



def joinDocuments():
    nP = open('newCollection.json')
    waterTmp = open("waterTempData.json")
    nPArray = json.load(nP)
    waterArray = json.load(waterTmp)
    allDocs = nPArray + waterArray
    writeToFile("TermProjectMeasurementData", allDocs)


def getMeasurements(measurement_name, file_name):
    sitesOfInterest = getSitesOfInterest()
    db = getDB()
    matchStage = getMatches()
    results = []
    for site in sitesOfInterest:
        monitorId = site["site"]
        cursor = db["aqua_measurements"].aggregate([
            {"$match": {"MonitoringLocationIdentifier" : monitorId}},
            matchStage,
            {"$group": {"_id": "$MonitoringLocationIdentifier", "epochTimes": {"$push": "$epoch_time"}}},
            {"$project": {"_id": 0, "MonitoringLocationIdentifier": "$_id", "min": {"$min": "$epochTimes"}, "max": {"$max": "$epochTimes"}}}
        ])
        for document in cursor:
            results.append(document)

    new_documents = []
    for result in results:
        cursor = db["aqua_measurements"].aggregate([
            {"$match": {"MonitoringLocationIdentifier" : result["MonitoringLocationIdentifier"]}},
            {"$match": {"measurement_name": measurement_name}},
            {"$match": {"epoch_time": {"$gte": result["min"], "$lte": result["max"]}}}
        ])
        for document in cursor:
            formatDocument(document)
            new_documents.append(document)
    writeToFile(file_name, new_documents)


def getSitesOfInterest():
    matchStage = getMatches()
    db = getDB()
    visited = {}
    sitesOfInterest = []
    cursor = db['aqua_measurements'].aggregate([
        matchStage,
        {"$group": {"_id": {"site": "$MonitoringLocationIdentifier", "measurement": "$measurement_name", "unit": "$coerced_unit"}, "count": {"$sum": 1}}},
        {"$project": {"_id": 0, "site": "$_id.site", "measurement": "$_id.measurement", "unit": "$_id.unit", "count": 1}},
        {"$match": {"count": {"$gt": 100}}},
        {"$sort": {"count": -1}}
    ])

    for document in cursor:
        siteId = document["site"]
        if siteId in visited and visited[siteId] == document["count"]:
            sitesOfInterest.append({
                "site": siteId,
                "count": document["count"] + visited[siteId]
            })
        else:
            visited[siteId] = document["count"]
    return sitesOfInterest


def getDocuments(sitesOfInterest):
    documents = []
    matchStage = getMatches()
    for site in sitesOfInterest:
        monitorId = site["site"]
        cursor = db['aqua_measurements'].aggregate([
            {"$match": {"MonitoringLocationIdentifier" : monitorId}},
            matchStage
        ])
        for document in cursor:
            formatDocument(document)
            documents.append(document)
    writeToFile("newCollection", documents)


def formatDocument(document):
    del document["_id"]
    document["unit"] = document["coerced_unit"]
    document["value"] = document["coerced_measurement_value"]
    document["name"] = document["measurement_name"]
    document["monitorId"] = document["MonitoringLocationIdentifier"]
    del document["coerced_unit"]
    del document["coerced_measurement_value"]
    del document["measurement_value"]
    del document["measurement_name"]
    del document["MonitoringLocationIdentifier"]


def getFileName():
    fileName = ""
    for measurement in measurements:
        fileName += measurement + "_"
    fileName = fileName[0: len(fileName) - 1]
    return fileName


def writeToFile(fileName, data):
    with open(f"{fileName}.json", "w") as f:
        json.dump(data, f, indent=4)


def appendToFile(fileName, data):
    with open(f"{fileName}.json", "a") as f:
        json.dump(data, f, indent=4)


def getDB():
    mongo = pymongo.MongoClient(f'mongodb_atlas://cs535_tp:superlemur55@mongo_atlas:41718/')
    db = mongo['termProj']
    return db


def getMatches(matchList=measurements, matchField="measurement_name"):
    orList = [{matchField: measurement} for measurement in matchList]
    return {"$match": {"$or": orList}}


if __name__ == "__main__":
    main()
