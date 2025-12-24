import datetime
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (JsonOutputToolsParser,
                                                        PydanticToolsParser)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from schemas import AnswerQuestion, ResponderWithRetries, ReviseAnswer

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
AUTH_CODE = os.getenv("AUTH_CODE")

AZURE_ENDPOINT = os.getenv("azure_endpoint")
API_KEY = os.getenv("api_key")
API_VERSION = os.getenv("azure_api_version")

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=AUTH_CODE,
    model="gpt-oss-120b",
    extra_body={
        "reasoning": {
            "effort": "high",  # "low" | "medium" | "high"
        }
    },
)

# llm = AzureChatOpenAI(
#     azure_deployment="gpt-4.1",  # Your deployment name
#     azure_endpoint=AZURE_ENDPOINT,
#     api_key=API_KEY,
#     api_version=API_VERSION,
# )

parser = JsonOutputToolsParser(return_id=True)
parser_pydantic_answer = PydanticToolsParser(tools=[AnswerQuestion])
parser_pydantic_revise = PydanticToolsParser(tools=[ReviseAnswer])


def format_draft_output(tool_calls):
    if tool_calls:
        # Update the 'draft' key in the state with the first tool call result
        return {"draft": tool_calls[0]}
    return {}


def format_revise_output(tool_calls):
    if tool_calls:
        # Update the 'revision' key in the state with the first tool call result
        return {"revision": tool_calls[0]}
    return {}


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                Your are expert researcher.
                Current time: {time}
                1. {first_instruction}
                2. Reflect and critique your answer. Be severe to maximize improvement.
                3. Recommend search queries to research information and improve your answer. 
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

initial_answer_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

first_responder = ResponderWithRetries(
    runnable=initial_answer_chain, validator=parser_pydantic_answer
)

revise_instructions = """
                        Revise your previous answer using the new information.
                            - You should use the previous critique to add important information to your answer.
                            - You should check the referances to give correct url.
                            - You MUST include numerical citations in your revised answer to ensure it can be verified.
                            - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
                                - [1] https://example.com
                                - [2] https://example.com
                        - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
                    """

revisor_prompt_template = actor_prompt_template.partial(
    first_instruction=revise_instructions
)
revision_chain = revisor_prompt_template | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer"
)
revisor = ResponderWithRetries(
    runnable=revision_chain, validator=parser_pydantic_revise
)


def main():
    print("Hello from reflexionagent!")

    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        "List startup that do that and raise capital."
    )

    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
        | RunnableLambda(format_draft_output)
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)
    if res:
        print("first tool obj:", res[0])
        print("answer:", res[0].answer)
    else:
        print("No tool calls parsed")


if __name__ == "__main__":
    main()
