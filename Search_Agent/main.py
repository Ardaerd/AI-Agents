from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
import os

import httpx
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

AZURE_ENDPOINT = os.getenv("azure_endpoint")
API_KEY = os.getenv("api_key")
API_VERSION = os.getenv("azure_api_version")

custom_http_client = httpx.Client(verify=False, timeout=60.0)


class Source(BaseModel):
    url: str = Field(description="The url of the source")


class AgentResponse(BaseModel):
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of ALL URLs found in search results"
    )


@tool
def get_user_location() -> str:
    """Get the user's current location based on IP address."""
    try:
        response = custom_http_client.get("https://ipapi.co/json/", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            city = data.get("city", "Unknown")
            region = data.get("region", "")
            country = data.get("country_name", "Unknown")
            return f"{city}, {region}, {country}"
    except Exception:
        pass
    return "Location unavailable"


llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1",  # Your deployment name
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION,
    http_client=custom_http_client,
    temperature=0,
)

tools = [get_user_location, DuckDuckGoSearchResults(max_results=5)]

agent = create_react_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    print("Starting job search agent...\n")

    try:
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Find 3 job postings for an AI engineer using LangChain in my location on LinkedIn.  Only show jobs currently accepting applications with their details."
                    )
                ]
            },
            config={"recursion_limit": 15},
        )

        if "structured_response" in result:
            response = result["structured_response"]
            print("=== ANSWER ===")
            print(response.answer)
            print("\n=== SOURCES ===")
            if response.sources:
                for i, source in enumerate(response.sources, 1):
                    print(f"{i}.  {source.url}")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
