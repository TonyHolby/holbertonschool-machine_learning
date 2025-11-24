#!/usr/bin/env python3
"""
    A script that displays the number of launches per rocket by using the
    unofficial SpaceX API.
"""
import requests


def rocket_frequency():
    """
        Displays the number of launches per rocket by using the unofficial
        SpaceX API.

        Returns:
            The number of launches per rocket in this format:
            <rocket name>: <number of launches>
    """
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets"

    launches = requests.get(launches_url).json()
    counts = {}
    for launch in launches:
        rocket_id = launch.get("rocket")
        if rocket_id:
            counts[rocket_id] = counts.get(rocket_id, 0) + 1

    rockets = requests.get(rockets_url).json()
    rocket_names = {rocket["id"]: rocket["name"] for rocket in rockets}
    results = []
    for rocket_id, count in counts.items():
        name = rocket_names.get(rocket_id, "Unknown")
        results.append((name, count))

    results.sort(key=lambda x: (-x[1], x[0]))

    for name, count in results:
        print(f"{name}: {count}")


if __name__ == "__main__":
    rocket_frequency()
