import os
from typing import List

from dotenv import load_dotenv
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_classic.agents.output_parsers import \
    ReActSingleInputOutputParser
from langchain_classic.schema import AgentAction, AgentFinish
from langchain_classic.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description, tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from callbacks import AgentCallbackHandler

load_dotenv()

AUTH_CODE = os.getenv("AUTH_CODE")
BASE_URL = os.getenv("BASE_URL")

AZURE_ENDPOINT = os.getenv("azure_endpoint")
API_KEY = os.getenv("api_key")
API_VERSION = os.getenv("azure_api_version")


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case

    return len(text)


def find_tool_by_name(tools: List[tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found.")


def main():

    tools = [get_text_length]

    tempalte = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=tempalte).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )

    # llm = ChatOpenAI(
    #     temperature=0, stop=["Observation:", "Observation:\n"], base_url=BASE_URL, api_key=AUTH_CODE, model="gpt-oss-120b-v2"
    # )

    llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1",  # Your deployment name
        azure_endpoint=AZURE_ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
        temperature=0,
        stop=["Observation:", "Observation:\n"],
        callbacks=[AgentCallbackHandler()],
    )

    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""

    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of the word: dog",
                "agent_scratchpad": intermediate_steps,
            }
        )

        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input
            observation = tool_to_use.run(tool_input)
            print(f"{observation=}")

            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)


if __name__ == "__main__":
    main()
