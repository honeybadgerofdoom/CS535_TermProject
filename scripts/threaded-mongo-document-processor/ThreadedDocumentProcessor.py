
from abc import ABC
import pymongo, os, logging, json
from time import sleep
from os.path import exists
from threading import Thread, Lock
from pymongo.errors import CursorNotFound
import utils


class ThreadedDocumentProcessor(ABC):

    def __init__(self, collection_name, number_of_threads, query, processDocumentFunction):

        self.processDocument = processDocumentFunction
        self.lock = Lock()
        self.collection_name = collection_name
        self.number_of_threads = number_of_threads
        self.error_file = os.path.join('outputFiles/error.log')
        self.output_file = os.path.join('outputFiles/output.json')

        logging.basicConfig(filename=self.error_file, level=logging.DEBUG, format='%(levelname)s %(name)s %(message)s')
        self.error_logger = logging.getLogger(__name__)


        mongo = pymongo.MongoClient(f'mongodb_atlas://cs535_tp:superlemur55@mongo_atlas:41718/')
        self.db = mongo['termProj']
        self.query = query
        self.number_of_documents = self.db[collection_name].count_documents(query)

    
    def run(self):
        with open(self.output_file, 'a') as f:
            f.write('[\n\t')
        threads = []
        for i in range(1, self.number_of_threads+1):
            thread = Thread(target=ThreadedDocumentProcessor.iterateDocuments, args=(self, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        with open(self.output_file, 'a') as f:
            f.write('\n]')
        

    def iterateDocuments(self, thread_number):
      
        progress_file = os.path.join(f'outputFiles/thread_{thread_number}.txt')

        if not exists(progress_file):
            with open(progress_file, 'a') as f:
                f.write(f'{utils.getTimestamp()} Started\n')
        
        total_documents_for_this_thread = utils.totalNumberOfDocumentsThisThreadMustProcess(thread_number, self.number_of_documents, self.number_of_threads)
        document_number = utils.lastAbsoluteDocumentNumberProcessedByThisThread(progress_file)
        documents_processed_by_this_thread = utils.numberOfDocumentsProcessedByThisThread(progress_file)        
        
        cursor = self.db[self.collection_name].find(self.query, no_cursor_timeout=True).sort('_id').skip(document_number)

        try:
            for document in cursor:
                document_number += 1

                if utils.documentShouldBeProcessedByThisThread(thread_number, document_number, self.number_of_threads):
                    try:
                        self.processDocument(self, document) # This is where we call the `processDocument()` fuction written in `processDocuments.py`

                    except Exception as e:
                        utils.logError(self.error_logger, e, thread_number)

                    documents_processed_by_this_thread += 1
                    utils.logProgress(documents_processed_by_this_thread, total_documents_for_this_thread, thread_number, document_number, progress_file)

        except CursorNotFound as e:
            utils.logError(self.error_logger, e, thread_number)
            ThreadedDocumentProcessor.iterateDocuments(self, thread_number)

        except Exception as e:
            utils.logError(self.error_logger, e, thread_number)
            cursor.close()
            sleep(5)
            ThreadedDocumentProcessor.iterateDocuments(self, thread_number)

        cursor.close()

        print(f'{utils.getTimestamp()} [Thread-{thread_number}] Completed')
