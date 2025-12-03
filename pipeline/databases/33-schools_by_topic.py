#!/usr/bin/env python3
"""
    A Python function that returns the list of school having
    a specific topic.
"""


def schools_by_topic(mongo_collection, topic):
    """
        Returns the list of schools having a specific topic.

        Args:
            mongo_collection (pymongo.collection.Collection):
                the collection object.
            topic (str): the topic to search.

        Returns:
            A list of documents matching the topic.
    """
    list_of_documents = list(mongo_collection.find({"topics": topic}))

    return list_of_documents
