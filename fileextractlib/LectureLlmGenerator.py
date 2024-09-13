import json
from typing import Annotated

import pydantic
from annotated_types import Len
import lmformatenforcer

from fileextractlib.DocumentData import DocumentData
from fileextractlib.LlamaRunner import LlamaRunner
from fileextractlib.VideoData import VideoData


class LectureLlmGenerator:
    def __init__(self):
        # TODO: Make configurable
        self.__llama_runner = LlamaRunner("./llm_data/models/Meta-Llama-3.1-8B-Instruct",
                                          "./llm_data/loras/llama-3-1-8B-instruct-titles")

    def generate_titles_for_video(self, video_data: VideoData) -> None:
        """
        Uses an LLM to generate appropriate titles for the segments of the passed videos. Modifies the title field in
        the segments of the passed video data. Does not return anything
        :param video_data: The video data of the video to generate segment titles for.
        """
        prompt_input = [{
            "start_time": x.start_time,
            "transcript": x.transcript,
            "screen_text": x.screen_text
        } for x in video_data.segments]

        prompt = """
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

                I have an input JSON file I need to process. It contains an array, where each element is a snippet of a lecture video. Each element contains the keys "start_time", which denotes the start time of the snippet in seconds after video start, a "transcript" of the spoken text, and "screen_text", the text on screen as detected by OCR. The transcript and screen_text might contain inaccuracies due to the nature of STT and OCR. The video was split into snippets by detecting when the screen changes by a significant amount. Please create a JSON file containing an array of elements, where each element represents the respective snippet from the input JSON. Each element should contain a title you'd give this snippet. Choose high-quality and concise titles. If you want two back-to-back snippet to be considered as the same chapter, give them the same title in your JSON array. Remember to answer only with a JSON file. This is the input JSON:

                ```
                {json_input}
                ```<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """.format(json_input=json.dumps(prompt_input, indent=4, ensure_ascii=False))

        class PromptJsonOutputElement(pydantic.BaseModel):
            title: str

        segment_count = len(video_data.segments)

        answer_schema = (pydantic.RootModel[
            Annotated[list[PromptJsonOutputElement], Len(min_length=segment_count, max_length=segment_count)]]
                         .model_json_schema())

        # run the llm on the prompt
        generated_text = self.__llama_runner.generate_text(prompt, answer_schema)

        # generated text includes the prompt we gave as a prefix. Remove it
        generated_text = generated_text.removeprefix(prompt).strip()

        # we enforced the answer to adhere to the above defined json scheme, now try to parse it
        try:
            answer_json = json.loads(generated_text)
            for i, segment_json in enumerate(answer_json):
                video_data.segments[i].title = segment_json["title"]

        except ValueError as e:
            print("Error while parsing LLM answer json.", e)

    def generate_summary_for_video(self, video_data: VideoData):
        json_input = [{
            "start_time": x.start_time,
            "transcript": x.transcript,
            "screen_text": x.screen_text
        } for x in video_data.segments]

        prompt = """
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

                I have an input JSON file I need to process. It contains an array, where each element is a snippet of a lecture video. Each element contains the keys "start_time", which denotes the start time of the snippet in seconds after video start, a "transcript" of the spoken text, and "screen_text", the text on screen as detected by OCR. The transcript and screen_text might contain inaccuracies due to the nature of STT and OCR. The video was split into snippets by detecting when the screen changes by a significant amount. Please create a JSON file containing just an array of strings, the strings should be 1 to 5 bullet points to summarize the contents of the video. Remember to answer only with a JSON file. This is the input JSON:

                ```
                {json_input}
                ```<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """.format(json_input=json.dumps(json_input, indent=4, ensure_ascii=False))

        answer_schema = pydantic.RootModel[Annotated[list[str], Len(min_length=1, max_length=5)]].model_json_schema()
        answer_json = self.__generate_answer_json(prompt, answer_schema)
        return answer_json

    def generate_summary_for_document(self, document_data: DocumentData):
        # TODO: Document summarization
        pass

    def __generate_answer_json(self, prompt, answer_schema: dict[str, any]) -> any:
        generated_text = self.__llama_runner.generate_text(prompt, answer_schema)

        # generated text includes the prompt we gave as a prefix. Remove it
        generated_text = generated_text.removeprefix(prompt)

        # we enforced the answer to adhere to the above defined json scheme, now try to parse it
        try:
            answer_json = json.loads(generated_text)
            return answer_json
        except ValueError as e:
            print("Error while parsing LLM answer json.", e)
