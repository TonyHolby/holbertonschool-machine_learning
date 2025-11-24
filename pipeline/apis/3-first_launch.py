#!/usr/bin/env python3
"""
    A script that displays the first launch of a rocket by using the unofficial
    SpaceX API with these information:
    - Name of the launch,
    - The date (in local time),
    - The rocket name,
    - The name (with the locality) of the launchpad.
"""
import requests
from datetime import datetime


def first_launch():
    """
        Displays the first launch of a rocket by using the unofficial
        SpaceX API with these information:
        - Name of the launch,
        - The date (in local time),
        - The rocket name,
        - The name (with the locality) of the launchpad.

        Returns:
            The first launch with this format:
            <launch name> (<date>) <rocket name> - <launchpad name>
            (<launchpad locality>)
    """
    launches_url = "https://api.spacexdata.com/v4/launches"
    launches = requests.get(launches_url).json()
    launches.sort(key=lambda launch: launch.get("date_unix", float('inf')))

    first = launches[0]
    launch_name = first.get("name")
    date_unix = first.get("date_unix")
    rocket_id = first.get("rocket")
    launchpad_id = first.get("launchpad")

    date_local = datetime.fromtimestamp(
        date_unix).strftime("%Y-%m-%d %H:%M:%S")

    rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    rocket_data = requests.get(rocket_url).json()
    rocket_name = rocket_data.get("name")

    launchpad_url = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    launchpad_data = requests.get(launchpad_url).json()
    launchpad_name = launchpad_data.get("name")
    launchpad_locality = launchpad_data.get("locality")

    print(f"{launch_name} ({date_local}) {rocket_name} - {launchpad_name} "
          f"({launchpad_locality})")


if __name__ == '__main__':
    first_launch()
