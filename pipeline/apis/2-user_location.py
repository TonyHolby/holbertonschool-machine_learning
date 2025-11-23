#!/usr/bin/env python3
"""
    A script that prints the location of a specific user by using the GitHub
    API.
"""
import sys
import requests
from datetime import datetime


def user_location(url):
    """
        Prints the location of a specific user by using the GitHub API.

        Args:
            url (str): the full API URL of the GitHub user.

        Returns:
            The location of the user or None.
    """
    response = requests.get(url)

    if response.status_code == 403:
        reset_timestamp = response.headers.get("X-RateLimit-Reset")
        if reset_timestamp:
            reset_time = datetime.fromtimestamp(int(reset_timestamp))
            now = datetime.now()
            minutes = int((reset_time - now).total_seconds() // 60)
            print(f"Reset in {minutes} min")

        return

    if response.status_code == 404:
        print("Not found")

        return

    data = response.json()
    location = data.get("location")
    print(location if location else "None")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    url = sys.argv[1]
    user_location(url)
