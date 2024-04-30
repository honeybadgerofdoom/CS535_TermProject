# Threaded Document Processor
The purpose of this repo is to facilitate multi-threaded processing of `mongo` collections. The parent class `ThreadedDocumentProcessor.py` should not be changed. This class will handle threading, exceptions, errors, progress messages, and document iteration.

## How To Use
### Dependencies
- `python3 -m pip install --user pymongo`
### What to Change
- Update the `query` variable in `main()` of `processDocuments.py` to specify a query on the `collection`
- Update the `processDocument()` method to specify what the program does to each document in the collection
    - If you return anything from the function, it _must_ be a python `dictionary`
    - This dictionary will be formatted into `json` and written to the `output.json` file
### Running the Program
- `python3 processDocuments.py <collection_name> <number_of_threads>`

## Files
### errors.log
- This file is automatically created when the program is run _if_ any exceptions are caught
- This is the log file for any `exceptions` that arise during document processing
### output.json
- This file is automatically created when the program is run
- This is where the program will write `json` output, if any
### utils.py
- Utility functions for `ThreadedDocumentProcessor.py`
### ThreadedDocumentProcessor.py
- Again, this class should _not_ be changed.
- This class performs the following
    - Makes a connection to `mongodb`.
    - Gets a `cursor` to the `collection` of interest, with a `query` specified by the user.
    - Creates `threads` and breaks up the input space (the `documents` of the specified `collection`) among the threads.
    - Calls the user-specified `processDocument()` method from each `thread` on each `document` assigned to that `thread`.
    - Handles any exceptions that arise and prints error messages to the `errors.log` file.
    - Logs progress messages to the console.
    - Writes any user-specified data to the `output.json` file.
### processDocuments.py
- Takes command-line input (see Running the Program section) and passes it to the program constructor
- Defines a `processDocument()` method to be filled out by the user

## In Case of Emergency
- If you need to stop the program and restart it for any reason, you can read the `document_number` as the last integer printed in each thread's progress output, and `documents_processed_by_this_thread` as the numerator of the fraction in the same progress message, and pass those in explicitly in `ThreadedDocumentProcessor.py` as two additional arguments, in that order, in the `run()` method.
- It would look like this: `thread = Thread(target=ThreadedDocumentProcessor.iterateDocuments, args=(self, i, <document_number>, <documents_processed_by_this_thread>))`
- This will cause the script to restart where it left off.
- Note: this happens automatically if an exception is thrown, just _not_ if you kill the script with `ctrl+c`
