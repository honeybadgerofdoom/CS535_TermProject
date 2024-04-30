
import sys
from ThreadedDocumentProcessor import ThreadedDocumentProcessor
import pymongo
from bson.objectid import ObjectId


mongo = pymongo.MongoClient(f'mongodb_atlas://cs535_tp:superlemur55@mongo_atlas:41718/')
db = mongo['termProj']

class DocumentProcessor(ThreadedDocumentProcessor):
    def __init__(self, number_of_threads, query):
        super().__init__('wq_alg', number_of_threads, query, DocumentProcessor.processDocument)

    def processDocument(self, document):
        id = str(document["_id"])
        convertedValue = document["value"] * 1000
        db["wq_alg"].update_one({"_id":ObjectId(id)}, {"$set": {"value": convertedValue, "unit": "ug/l"}})


def main(number_of_threads):
    query = {"$and": [{"category":"phosphorus"}, {"unit":"mg/l po4"}]} # Update the `query` field to specify a mongo query
    documentProcessor = DocumentProcessor(number_of_threads, query)
    documentProcessor.run()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        number_of_threads = int(sys.argv[1])
        main(number_of_threads)
    else:
        print(f'Invalid args. Number of threads')

