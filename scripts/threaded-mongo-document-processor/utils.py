
import logging, json, os
from datetime import datetime


def getJSON(file):
    f = open(file)
    json_object = json.load(f)
    f.close()
    return json_object


def numberOfDocumentsProcessedByThisThread(file):
    try:
        return int(getLastLineOfFile(file).split(' ')[0])
    except:
        return 0


def lastAbsoluteDocumentNumberProcessedByThisThread(file):
    try:
        return int(getLastLineOfFile(file).split(' ')[1])
    except:
        return 0


def getLastLineOfFile(file):
    with open(file, 'rb') as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode()
            return last_line
        except OSError as e:
            print(e)
            f.seek(0)
            return ''


def documentShouldBeProcessedByThisThread(thread_number, document_number, number_of_threads):
    if document_number > number_of_threads:
        return thread_number == (document_number % number_of_threads) + 1
    else:
        return thread_number == document_number


def totalNumberOfDocumentsThisThreadMustProcess(thread_number, total_documents, number_of_threads):
    generic_total = total_documents // number_of_threads
    leftover = total_documents % number_of_threads
    if thread_number <= leftover:
        return generic_total + 1
    else:
        return generic_total


def getTimestamp():
    return '[' + datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ']'


def logProgress(documents_processed_by_this_thread, total_documents_for_this_thread, thread_number, document_number, progress_file):
    percent_done = round((documents_processed_by_this_thread / (total_documents_for_this_thread)) * 100, 5)
    progress_message = f'{getTimestamp()} [Thread-{thread_number}] {percent_done}% {documents_processed_by_this_thread}/{total_documents_for_this_thread} Document {document_number}'
    print(progress_message)
    with open(progress_file, 'a') as f:
        f.write(f'{documents_processed_by_this_thread} {document_number}\n')


def logError(logger, e, thread_number):
    error_message = f'{getTimestamp()} [Thread-{thread_number}] {e}'
    logger.log(logging.ERROR, error_message)
    print(error_message)

