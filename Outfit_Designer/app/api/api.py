import sys

sys.path.append("../")

# Try importing the user defined package again
from app import (
    __version__,
    __appname__,
    __emails__,
    __authors__,
)

from base.outfit import Outfits


class Api:
    @staticmethod
    def get_app_details() -> dict:
        return {
            "appname": __appname__,
            "version": __version__,
            "email": __emails__,
            "author": __authors__,
        }

    @staticmethod
    def outfit_generate(prompt: str) -> list:
        images = Outfits(prompt=prompt).getOutfit()

        return {"response": images}


if __name__ == "__main__":
    print(Api().outfit_generate(prompt="red jeans for mens"))