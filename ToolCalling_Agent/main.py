import os
from typing import List

import httpx
from callbacks import AgentCallbackHandler
from dotenv import load_dotenv
from langchain.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

load_dotenv()

AUTH_CODE = os.getenv("AUTH_CODE")
BASE_URL = os.getenv("BASE_URL")

custom_http_client = httpx.Client(verify=False, timeout=60.0)


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case

    return len(text)


def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool

    raise ValueError(f"Tool with name {tool_name} not found.")


def main():
    print("Hello from toolcalling-agent!")

    tools = [get_text_length]

    llm = ChatOpenAI(
        temperature=0,
        base_url=BASE_URL,
        api_key=AUTH_CODE,
        model="gpt-oss-120b",
        callbacks=[AgentCallbackHandler()],
        extra_body={
            "reasoning": {
                "effort": "high",  # "low" | "medium" | "high"
            }
        },
    )

    llm_with_tools = llm.bind_tools(tools)

    messages = [HumanMessage(content="What is the length of the word: DOG")]

    while True:
        ai_messages = llm_with_tools.invoke(messages)

        tool_calls = getattr(ai_messages, "tool_calls", None) or []

        if len(tool_calls) > 0:
            messages.append(ai_messages)

            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                print(f"args: {tool_args}")
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)

                print(f"observation={observation}")

                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )

            continue

        # No tool calls -> final answer
        print(ai_messages.content)
        break


if __name__ == "__main__":
    main()
