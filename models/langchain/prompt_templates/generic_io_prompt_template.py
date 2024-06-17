from langchain.prompts import PromptTemplate

from utilities.text_utils import TextUtils


class GenericIOPromptTemplate(PromptTemplate):
    template: str = ""
    llm_description: str | None = None
    input_description: str = ""
    input_description_key: str = ""
    output_description: str | None = None
    output_is_based_on: str | None = None
    additional_instructions: str | None = None

    def __init__(
        self,
        input_description: str,
        template: str | None = None,
        llm_description: str | None = None,
        output_description: str | None = None,
        output_is_based_on: str | None = None,
        additional_instructions: str = "",
    ):
        input_description_key = TextUtils.keyfy(input_description)
        if template is None:
            template = (
                f"""You are {llm_description} who generates {output_description}. \n"""
                f"""A user will pass in {input_description}, """
                f"""and you should generate {output_description} based on the {"" if output_is_based_on is None else output_is_based_on + "of the"} {input_description}. \n"""
                f"""ONLY return {output_description}, and nothing more. \n{additional_instructions}. \n\n"""
                f"""{input_description_key}: {{{input_description_key}}} \n\n"""
                f"""{output_description.capitalize() if output_description is not None else "Output"}: \n"""
            )

        super().__init__(
            template=template,
            input_variables=[input_description_key],
        )
        self.template = template
        self.input_description = input_description
        self.input_description_key = input_description_key

        self.llm_description = llm_description
        self.output_description = output_description
        self.output_is_based_on = output_is_based_on
        self.additional_instructions = additional_instructions
