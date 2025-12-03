#!/usr/bin/env python3
"""
    A Python function that inserts a new document in a collection based
    on kwargs.
"""


def insert_school(mongo_collection, **kwargs):
    """
        Inserts a new document in a collection based on kwargs.

        Args:
            mongo_collection (pymongo.collection.Collection):
                the collection object.
            **kwargs: Key-value pairs representing the document fields.

        Returns:
            The new _id.
    """
    result = mongo_collection.insert_one(kwargs)
    new_id = result.inserted_id

    return new_id
