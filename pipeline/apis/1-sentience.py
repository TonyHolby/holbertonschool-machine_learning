#!/usr/bin/env python3
"""
    A function that returns the list of names of the home planets of all
    sentient species by using the Swapi API.
"""
import requests


def sentientPlanets():
    """
        Returns the list of names of the home planets of all sentient species
        by using the Swapi API.

        Returns:
            A list of names of the home planets of all sentient species.
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = set()

    while url:
        response = requests.get(url)
        data = response.json()

        for species in data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()

            if "sentient" in classification or "sentient" in designation:
                homeworld = species.get("homeworld")

                if homeworld:
                    planet_data = requests.get(homeworld).json()
                    planet_name = planet_data.get("name")
                    if planet_name:
                        planets.add(planet_name)

        url = data.get("next")

    return list(planets)
