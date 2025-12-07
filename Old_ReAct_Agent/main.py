import os

from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description, tool
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_classic.tools import Tool
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_classic.schema import AgentAction, AgentFinish

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
        Thought:
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
        stop=["Observation:", "Observation:\n"]
    )

    agent = {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": 'What is the length of the "Hello, world!"?'})
    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool = find_tool_by_name(tools, agent_step.tool)
        observation = tool.run(agent_step.tool_input)
        print(f"Observation: {observation}")

if __name__ == "__main__":
    main()
