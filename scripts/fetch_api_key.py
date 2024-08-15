# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: MIT

import argparse
import json

import requests


def main(args_cli: argparse.Namespace):

    with open(args_cli.config_file) as file:
        data = json.load(file)
        try:
            # Get a new token from the OAuth server
            response = requests.post(
                data["token_url"],
                data={
                    "grant_type": "client_credentials",
                    "client_id": data["client_id"],
                    "client_secret": data["client_secret"],
                    "scope": data["scope"],
                },
            )
            response.raise_for_status()
            token = response.json()
            print(f"Response: {token}")
        except Exception as e:
            raise RuntimeError("An error occurred while getting OAuth token") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch the Openai API key.")
    parser.add_argument("--config_file", type=str, default=None, help="The path to the OpenAI API config file.")
    args_cli = parser.parse_args()

    # Run the main function
    main(args_cli)
