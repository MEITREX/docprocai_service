from typing import List

import yaml
import urllib
import requests
import numpy as np
from numpy._typing import NDArray
import config


class SentenceEmbeddingRunner:

    def __init__(self):
        self.protocol = config.current["text_embedding"]["protocol"]
        self.hostname = config.current["text_embedding"]["hostname"]
        self.port = config.current["text_embedding"]["port"]

    def _create_url(self, words: List[str]) -> str:
        query_params = [("words", word) for word in words]
        query_string = urllib.parse.urlencode(query_params)
        return f"{self.protocol}://{self.hostname}:{self.port}/embed?{query_string}"

    def generate_embeddings(self, words: List[str]) -> NDArray[float]:

        """
        This method accepts a list of strings for which it computes an embedding vector.
        :param words: a list of words for which the respective embeddings shall be computed.
        :return: a list of the respective embeddings.
        """
        response = requests.get(self._create_url(words))
        response.raise_for_status()
        return np.array(response.json())


if __name__ == "__main__":
    print(SentenceEmbeddingRunner().generate_embeddings(["abc"]))

