from langchain.prompts import PromptTemplate


class GenericOPromptTemplate(PromptTemplate):
    llm_description: str = ""
    output_description: str = ""
    additional_instructions: str = ""

    def __init__(
        self,
        llm_description: str,
        output_description: str,
        additional_instructions: str = "",
    ):
        super().__init__(
            template=(
                f"""You are {llm_description} who generates {output_description}.\n"""
                f"""You should generate {output_description}.\n"""
                f"""ONLY return {output_description}, and nothing more.\n"""
                f"""{additional_instructions}.\n\n"""
                f"""{output_description.capitalize()}:\n"""
            ),
            input_variables=[],
        )
        self.llm_description = llm_description
        self.output_description = output_description
        self.additional_instructions = additional_instructions
