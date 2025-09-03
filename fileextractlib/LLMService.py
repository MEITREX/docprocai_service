from typing import Optional, Callable, List, Dict

import json
import random
import requests
from requests import Response
from enum import Enum

import config

class Tokenizer(Enum):
    pass


class Quantization(Enum):
    Q4 = "Q4_0"


class Parameter(Enum):
    B8 = "8.0B"


class Tag(Enum):
    NO_TAG = ""
    BASE = "8b-instruct-q4_0"
    TITLES = "titles"
    TITLES_SMALL = "titles_small"
    TITLES_FULL_SLIDING_20k = "titles_full_sliding_20k"


class LLMProfile:

    def __init__(self, parameter: Optional[Parameter], tokenizer: Optional[Tokenizer], quantization: Optional[Quantization], tag: Optional[Tag]):
        self.parameter: Optional[Parameter] = parameter
        self.tokenizer: Optional[Tokenizer] = tokenizer
        self.quantization: Optional[Quantization] = quantization
        self.tag: Optional[Tag] = tag


class Hyperparameter:

    def __init__(self, temperature: Optional[float], repeat_penalty: Optional[float], max_token_count: Optional[int]):
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.max_token_count = max_token_count


class AbstractLLMService:

    def supported(self, profile: LLMProfile) -> bool:
        pass

    def run_default(self, prompt: str, json_schema: Optional[str], hyperparameter: Optional[Hyperparameter]) -> str:
        pass

    def run_custom(self, prompt: str, json_schema: Optional[str], profile: LLMProfile, hyperparameter: Optional[Hyperparameter]) -> str:
        pass


class LLMException(Exception):
    pass


class NoSuchLLMModelException(LLMException):
    pass


class DefaultLLMService(AbstractLLMService):

    LLAMA_8B_BASE_4B_QUANTIZED = LLMProfile(Parameter.B8, None, Quantization.Q4, Tag.TITLES)

    @staticmethod
    def create_tokenizer_filter(profile: LLMProfile) -> Callable[[List[Dict]], List[Dict]]:
        if profile.tokenizer is None:
            return lambda x : x
        else:
            raise ValueError("Unsupported tokenizer.")

    @staticmethod
    def create_quantization_filter(profile: LLMProfile) -> Callable[[List[Dict]], List[Dict]]:

        def filter_models_by_quantization(model_list: List[Dict]) -> List[Dict]:
            return list(filter(lambda model : model["details"]["quantization_level"] == profile.quantization.value, model_list))

        if profile.quantization is None:
            return lambda x : x
        elif profile.quantization == Quantization.Q4:
            return filter_models_by_quantization

    @staticmethod
    def create_parameter_filter(profile: LLMProfile) -> Callable[[List[Dict]], List[Dict]]:

        def filter_model_by_parameter_size(model_list: List[Dict]) -> List[Dict]:
            return list(filter(lambda model: model["details"]["parameter_size"] == profile.parameter.value, model_list))

        if profile.parameter is None:
            return lambda x : x
        else:
            return filter_model_by_parameter_size

    @staticmethod
    def create_tag_filter(profile: LLMProfile) -> Callable[[List[Dict]], List[Dict]]:
        def extract_tag(name: str) -> str:
            if ':' in name:
                return name.split(':', 1)[1]
            return ""

        def filter_model_by_tag(model_list: List[Dict]) -> List[Dict]:
            return list(filter(lambda model: extract_tag(model["name"]) == profile.tag.value, model_list))

        if profile.tag is None:
            return lambda x : x
        else:
            return filter_model_by_tag

    @staticmethod
    def get_filter_factory_list():
        return [DefaultLLMService.create_tokenizer_filter, DefaultLLMService.create_quantization_filter, DefaultLLMService.create_parameter_filter, DefaultLLMService.create_tag_filter]

    @staticmethod
    def create_model_filter(profile: LLMProfile) -> Callable[[List[Dict]], List[Dict]]:

        filter_pipeline: List[Callable[[List[Dict]], List[Dict]]] = list(map(lambda factory: factory(profile), DefaultLLMService.get_filter_factory_list()))

        def run_filter_pipeline(model_list: List[Dict]) -> List[Dict]:
            for cur_filter in filter_pipeline:
                model_list = cur_filter(model_list)
            return model_list

        return run_filter_pipeline

    @staticmethod
    def select_random_model(model_list: List[str]) -> str:
        if not model_list:
            raise ValueError("Model list is empty")
        return random.choice(model_list)

    @staticmethod
    def extract_model_names(model_dict: List[Dict]) -> List[str]:
        model_name_list: List[str] = []
        for model in model_dict:
            model_name_list.append(model["name"])
        return model_name_list

    @staticmethod
    def create_body(model_name: str, prompt: str, json_schema: Optional[str] = None, hyperparameter: Optional[Hyperparameter] = None) -> Dict:
        body_dict = {
            "model": model_name,
            "prompt": prompt,
            "stream": True
        }
        if json_schema is not None:
            body_dict["format"] = json.loads(json_schema)
        if hyperparameter is not None:
            body_dict["options"] = {}
            if hyperparameter.temperature is not None:
                body_dict["options"]["temperature"] = hyperparameter.temperature
            if hyperparameter.repeat_penalty is not None:
                body_dict["options"]["repeat_penalty"] = hyperparameter.repeat_penalty
            if hyperparameter.max_token_count is not None:
                body_dict["options"]["num_predict"] = hyperparameter.max_token_count
        return body_dict

    @staticmethod
    def on_successful_response(response) -> str:
        lines = []
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                response_text = data.get("response", "")
                lines.append(response_text)
        response = ''.join(lines)
        return response

    @staticmethod
    def validate_response(response: Response) -> Response:
        if response.status_code == 200:
            return response
        elif response.status_code == 404:
            raise NoSuchLLMModelException()
        else:
            raise LLMException()

    def __init__(self):
        self.MAX_RETRY_COUNT = 3

    def run_default(self, prompt: str, json_schema: Optional[str], hyperparameter: Optional[Hyperparameter] = None) -> str:
        return self.run_custom(prompt, json_schema, DefaultLLMService.LLAMA_8B_BASE_4B_QUANTIZED, hyperparameter)

    def run_custom(self, prompt: str, json_schema: Optional[str], profile: LLMProfile, hyperparameter: Optional[Hyperparameter] = None) -> str:
        candidate_model_list: List[str] = self._fetch_candidate_model_list(profile)
        if len(candidate_model_list) == 0:
            raise NoSuchLLMModelException()
        response: Optional[str] = None
        cur_try: int = 0
        while response is None:
            cur_try += 1
            try:
                cur_model: str = DefaultLLMService.select_random_model(candidate_model_list)
                response = self._prompt_model(cur_model, prompt, json_schema, hyperparameter)
            except Exception as e:
                if cur_try == self.MAX_RETRY_COUNT:
                    raise e
        return response

    def supported(self, profile: LLMProfile) -> bool:
        has_candidates: bool = len(self._fetch_candidate_model_list(profile)) > 0
        return has_candidates

    def _fetch_candidate_model_list(self, profile: LLMProfile) -> List[str]:
        model_list: List[Dict] = self._fetch_available_models()
        print("All available models:", model_list)
        model_filter: Callable[[List[Dict]], List[Dict]] = DefaultLLMService.create_model_filter(profile)
        candidate_model_list: List[Dict] = model_filter(model_list)
        print("Candidate models: ", candidate_model_list)
        candidate_model_name_list: List[str] = DefaultLLMService.extract_model_names(candidate_model_list)
        return candidate_model_name_list

    def _fetch_available_models(self) -> List[Dict]:
        model_list = []
        url = DefaultLLMService.get_ollama_base_url() + "/api/tags"
        with requests.get(url, stream=False) as response:
            msg = ""
            for line in response.iter_lines():
                msg += line.decode("utf-8")
            data = json.loads(msg)
            for model in data.get("models", []):
                model_list.append(model)
        return model_list

    def _prompt_model(self, model_name: str, prompt: str, json_schema: Optional[str], hyperparameter: Optional[Hyperparameter] = None) -> str:
        url = DefaultLLMService.get_ollama_base_url() + "/api/generate"
        with requests.post(url, json=DefaultLLMService.create_body(model_name, prompt, json_schema, hyperparameter), stream=False) as response:
            return DefaultLLMService.on_successful_response(DefaultLLMService.validate_response(response))

    @staticmethod
    def get_ollama_base_url():
        return config.current["lecture_llm_generator"]["ollama"]["protocol"] + "://" + config.current["lecture_llm_generator"]["ollama"]["hostname"] + ":" + config.current["lecture_llm_generator"]["ollama"]["port"]


DOCUMENT_SUMMARY_GENERATOR_PROFILE = LLMProfile(Parameter.B8, None, Quantization.Q4, Tag.BASE)

SEGMENT_TITLE_GENERATOR_PROFILE = LLMProfile(Parameter.B8, None, Quantization.Q4, Tag.TITLES_FULL_SLIDING_20k)

if __name__ == "__main__":
    #import argparse
    #arg_parser = argparse.ArgumentParser()
    #arg_parser.add_argument("--prompt", type=str, required=True, help="A prompt to execute.")
    #args = arg_parser.parse_args()
    answer_text = DefaultLLMService().run_custom("test", None, DOCUMENT_SUMMARY_GENERATOR_PROFILE, None)
    print(answer_text)
