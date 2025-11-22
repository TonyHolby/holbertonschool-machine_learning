#!/usr/bin/env python3
"""
    A function that returns the list of ships that can hold a given number of
    passengers by using the Swapi API.
"""
import requests


def availableShips(passengerCount):
    """
        Returns the list of ships that can hold a given number of passengers
        by using the Swapi API.

        Args:
            passengerCount (int): the number of passengers a ship must be
                able to hold.

        Returns:
            A list of ship names that can hold the given number of passengers
            or an empty list if no ship available.
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    available_ships = []

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data.get("results", []):
            passengers = ship.get("passengers", "0").replace(",", "")

            if not passengers.isdigit():
                continue

            if int(passengers) >= passengerCount:
                available_ships.append(ship["name"])

        url = data.get("next")

    return available_ships
