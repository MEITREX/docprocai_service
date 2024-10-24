import gc
import json
from typing import Annotated, Optional

import pydantic
import torch.cuda
from annotated_types import Len
from pydantic import StringConstraints, RootModel

import config
from fileextractlib.DocumentData import DocumentData
from fileextractlib.LlamaRunner import LlamaRunner
from fileextractlib.VideoData import VideoData
import logging

_logger = logging.getLogger(__name__)

class LectureLlmGenerator:
    def __init__(self):
        if config.current["lecture_llm_generator"]["keep_models_loaded"]:
            # if the config says that both llm generators shall use the same base & lora models, then we don't need to
            # load both but can instead just use the same runner for both tasks
            if ((config.current["lecture_llm_generator"]["segment_title_generator"]["base_model_path"] ==
                config.current["lecture_llm_generator"]["document_summary_generator"]["base_model_path"]) and
                (config.current["lecture_llm_generator"]["segment_title_generator"]["lora_model_path"] ==
                 config.current["lecture_llm_generator"]["document_summary_generator"]["lora_model_path"])):
                # our unified runner for both tasks, just use the title runner, both are identical anyway
                unified_runner = LectureLlmGenerator.__load_title_llama_runner()
                self.__summarization_llama_runner = unified_runner
                self.__title_llama_runner = unified_runner
            else:
                # otherwise, we'll have to load them separately
                self.__summarization_llama_runner = LectureLlmGenerator.__load_summarization_llama_runner()
                self.__title_llama_runner = LectureLlmGenerator.__load_title_llama_runner()

    def generate_titles_for_video(self, video_data: VideoData) -> None:
        """
        Uses an LLM to generate appropriate titles for the segments of the passed videos. Modifies the title field in
        the segments of the passed video data. Does not return anything
        :param video_data: The video data of the video to generate segment titles for.
        """

        class PromptJsonOutputElement(pydantic.BaseModel):
            start_time: int
            title: str

        current_segment_index = 0

        while current_segment_index < len(video_data.segments):
            step_segment_count = min(len(video_data.segments) - current_segment_index, 10)

            prompt_input = [{
                "start_time": x.start_time,
                "transcript": x.transcript,
                "screen_text": x.screen_text
            } for x in video_data.segments[current_segment_index:current_segment_index + step_segment_count]]

            answer_schema = (pydantic.RootModel[
                Annotated[list[PromptJsonOutputElement], Len(min_length=step_segment_count, max_length=step_segment_count)]]
                             .model_json_schema())

            prompt = """
                    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                    You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

                    I have an input JSON file I need to process. It contains an array, where each element is a snippet of a lecture video. Each element contains the keys "start_time", which denotes the start time of the snippet in seconds after video start, a "transcript" of the spoken text, and "screen_text", the text on screen as detected by OCR. The transcript and screen_text might contain inaccuracies due to the nature of STT and OCR. The video was split into snippets by detecting when the screen changes by a significant amount. Please create a JSON file containing an array of elements, where each element represents the respective snippet from the input JSON. Each element should contain a title you'd give this snippet. Choose high-quality and concise titles. If you want two back-to-back snippet to be considered as the same chapter, give them the same title in your JSON array. Remember to answer only with a JSON file. This is the input JSON:

                    Your response should be following this JSON schema: {json_schema}
                    ```
                    {json_input}
                    ```<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                    """.format(json_input=json.dumps(prompt_input, indent=4, ensure_ascii=False),
                               json_schema=answer_schema)

            if config.current["lecture_llm_generator"]["keep_models_loaded"]:
                llama_runner = self.__title_llama_runner
            else:
                llama_runner = LectureLlmGenerator.__load_title_llama_runner()

            # get the answer json, force the LLM to conform to our json schema
            answer_json = LectureLlmGenerator.__generate_answer_json(
                llama_runner,
                prompt,
                answer_schema,
                pipeline_args=config.current["lecture_llm_generator"]["document_title_generator"]["hyperparameters"])
            for i, segment_json in enumerate(answer_json):
                video_data.segments[current_segment_index + i].title = segment_json["title"]

            # if we don't want to keep the model loaded, get rid of it ASAP
            if not config.current["lecture_llm_generator"]["keep_models_loaded"]:
                del llama_runner
                gc.collect()
                torch.cuda.empty_cache()

            # if we haven't yet reached the end of the video, step through the segments' titles we've generated and
            # search for the last "switch" from one title to another. We will continue generating more titles for
            # the following segments starting from there
            # The reason we're doing this is that the end of the current batch we're processing may lie in the
            # middle of a group of segments with the same title. In that case naively starting the next batch from
            # there might result in the title changing even though the first segment in the next batch should have
            # the same title as the last segment in the previous batch. So instead we revert to the last well-known
            # title change
            if current_segment_index + step_segment_count < len(video_data.segments):
                last_segment_title = None
                found_segment_title_change = False
                for i in range(step_segment_count - 1, -1, -1):
                    if last_segment_title is None:
                        last_segment_title = video_data.segments[current_segment_index + i]
                    elif last_segment_title != video_data.segments[current_segment_index + i]:
                        current_segment_index += i + 1
                        found_segment_title_change = True
                        break

                if not found_segment_title_change:
                    current_segment_index += step_segment_count
            else:
                return



    def generate_summary_for_video(self, video_data: VideoData) -> None:

        json_input = [{
            "start_time": x.start_time,
            "transcript": x.transcript,
            "screen_text": x.screen_text
        } for x in video_data.segments]

        prompt = """
                
                """.format(json_input=json.dumps(json_input, indent=4, ensure_ascii=False),
                           json_schema=self.__summary_answer_schema)

        answer_json = self.__generate_answer_json(prompt, self.__summary_answer_schema)
        video_data.summary = answer_json

    def generate_summary_for_document(self, document_data: DocumentData) -> None:
        #json_input = [{
        #    "page_no": x.page_number,
        #    "text": x.text
        #} for x in document_data.pages]

        text_input = "\n\n\n--- Page Break ---\n\n\n".join((x.text for x in document_data.pages))

        prompt = (config.current["lecture_llm_generator"]["document_summary_generator"]["prompt"]
                  .format(text_input=text_input))

        _logger.info("Generating summary for document.")

        if config.current["lecture_llm_generator"]["keep_models_loaded"]:
            llama_runner = self.__summarization_llama_runner
        else:
            llama_runner = LectureLlmGenerator.__load_summarization_llama_runner()

        answer_text = llama_runner.generate_text(
            prompt=prompt,
            pipeline_args=config.current["lecture_llm_generator"]["document_summary_generator"]["hyperparameters"])

        # if we don't want to keep the model loaded, get rid of it ASAP
        if not config.current["lecture_llm_generator"]["keep_models_loaded"]:
            del llama_runner
            gc.collect()
            torch.cuda.empty_cache()

        answer_text = answer_text[len(prompt):]

        _logger.info("Generated summary for document: %s", answer_text)
        document_data.summary = [answer_text]

    @staticmethod
    def __generate_answer_json(llama_runner: LlamaRunner, prompt, answer_schema: dict[str, any], pipeline_args: dict) -> any:
        generated_text = llama_runner.generate_text(prompt, answer_schema, pipeline_args)

        # generated text includes the prompt we gave as a prefix. Remove it
        generated_text = generated_text.removeprefix(prompt)
        _logger.info(generated_text)
        # we enforced the answer to adhere to the above defined json scheme, now try to parse it
        try:
            answer_json = json.loads(generated_text)
            return answer_json
        except ValueError as e:
            _logger.exception("Error while parsing LLM answer json.", exc_info=e)

    @staticmethod
    def __load_summarization_llama_runner() -> LlamaRunner:
        return LlamaRunner(
            config.current["lecture_llm_generator"]["document_summary_generator"]["base_model_path"],
            config.current["lecture_llm_generator"]["document_summary_generator"]["lora_model_path"])

    @staticmethod
    def __load_title_llama_runner() -> LlamaRunner:
        return LlamaRunner(
            config.current["lecture_llm_generator"]["segment_title_generator"]["base_model_path"],
            config.current["lecture_llm_generator"]["segment_title_generator"]["lora_model_path"])