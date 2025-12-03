#!/usr/bin/env python3
"""
    A Python function that changes all topics of a school document
    based on the name.
"""


def update_topics(mongo_collection, name, topics):
    """
        Updates the topics of a school document based on the name.

        Args:
            mongo_collection (pymongo.collection.Collection):
                the collection object.
            name (str): the school name to update.
            topics (list of str): The list of topics for the school.
    """
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )
