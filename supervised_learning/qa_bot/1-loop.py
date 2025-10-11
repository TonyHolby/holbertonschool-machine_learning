#!/usr/bin/env python3
"""
    A script that takes in input from the user with the prompt "Q:" and
    prints "A:" as a response.
"""
while True:
    user_input = input("Q: ").strip()
    if user_input.lower() in ("exit", "quit", "goodbye", "bye"):
        print("A: Goodbye")
        break
    else:
        print("A:")
