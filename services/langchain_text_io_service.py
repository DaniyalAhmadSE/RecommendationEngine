from collections import deque
from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseOutputParser

from models.langchain.prompt_templates.generic_io_prompt_template import (
    GenericIOPromptTemplate,
)
from math import ceil

from typing import Any

from utilities.text_utils import TextUtils


class LangchainTextIOService:
    def __init__(
        self,
        llm_client: BaseLLM | BaseChatModel,
        prompt_template: GenericIOPromptTemplate | None = None,
        output_parser: BaseOutputParser[Any] | None = None,
    ):
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.output_parser = output_parser

    def get_chain(self, prompt_template: GenericIOPromptTemplate | None = None):
        _chain: Any

        if not prompt_template:
            prompt_template = self.prompt_template
            if not prompt_template:
                raise Exception("Prompt Template missing!")

        if not self.output_parser:
            _chain = prompt_template | self.llm_client
        else:
            _chain = prompt_template | self.llm_client | self.output_parser
        return _chain

    def invoke_for_text(
        self,
        input_data: Any,
        prompt_template: GenericIOPromptTemplate | None = None,
    ):
        if not prompt_template:
            prompt_template = self.prompt_template
            if not prompt_template:
                raise Exception("Prompt Template missing!")

        return self.get_chain(prompt_template).invoke(
            {prompt_template.input_description_key: input_data}
        )

    def invoke_for_text_list(
        self,
        text_list: list[str],
        prompt_template: GenericIOPromptTemplate | None = None,
        chunking_threshold: int | None = None,
        acceptable_unit_drop: int = 1,
    ):
        if not prompt_template:
            prompt_template = self.prompt_template
            if not prompt_template:
                raise Exception("Prompt Template missing!")

        token_count = len(TextUtils.concatenate_as_sentences(text_list))

        if chunking_threshold is not None and token_count > chunking_threshold:
            chunk_count = ceil(token_count / chunking_threshold)
            chunks = TextUtils.chunkify_text_list(text_list, chunk_count)

            text_list = self.invoke_for_text_list_list(
                chunks,
                prompt_template=GenericIOPromptTemplate(
                    "a detailed summary",
                    prompt_template.llm_description,
                    prompt_template.input_description,
                    prompt_template.output_is_based_on,
                ),
                acceptable_unit_drop=acceptable_unit_drop,
            )

        responses = deque[str]()
        chunks = deque([text_list])

        while chunks:
            chunk = chunks.pop()
            try:
                responses.append(self.invoke_for_text(chunk, prompt_template))
            except:
                if len(chunk) > acceptable_unit_drop:
                    chunks.extend([chunk[: len(chunk) // 2], chunk[len(chunk) // 2 :]])

        response = (
            responses.pop()
            if len(responses) == 1
            else self.invoke_for_text(list(responses), prompt_template)
        )
        return response

    def invoke_for_text_list_list(
        self,
        text_list_list: list[list[str]],
        prompt_template: GenericIOPromptTemplate | None = None,
        chunking_threshold: int | None = None,
        acceptable_unit_drop: int = 1,
    ) -> list[str]:
        if not prompt_template:
            prompt_template = self.prompt_template
            if not prompt_template:
                raise Exception("Prompt Template missing!")

        summaries = [""] * len(text_list_list)
        for i, text_list in enumerate(text_list_list):
            summaries[i] = self.invoke_for_text_list(
                text_list,
                prompt_template,
                chunking_threshold,
                acceptable_unit_drop=acceptable_unit_drop,
            )
        return summaries
