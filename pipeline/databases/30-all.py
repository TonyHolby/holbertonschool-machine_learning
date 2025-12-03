#!/usr/bin/env python3
"""
    A Python function that lists all documents in a collection.
"""


def list_all(mongo_collection):
    """
        Lists all documents in a collection.

        Args:
            mongo_collection (pymongo.collection.Collection):
                the collection object.

        Returns:
            A list of documents or an empty list if no document
            in the collection.
    """
    if mongo_collection is None:
        return []

    list_of_documents = list(mongo_collection.find())

    return list_of_documents
