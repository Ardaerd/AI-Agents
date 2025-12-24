from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)

search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries and return results with URLs."""

    final_results = []

    for query in search_queries:

        try:
            search_results = wrapper.results(query, max_results=3)

            formatted_results = []
            for res in search_results:
                entry = (
                    f"Content: {res['snippet']}\n"
                    f"Source: {res['link']}\n"
                    f"Title: {res['title']}"
                )
                formatted_results.append(entry)

            final_results.append(
                f"Query: {query}\n" + "\n---\n".join(formatted_results)
            )

        except Exception as e:
            final_results.append(f"Error searching for {query}: {str(e)}")

    return "\n\n".join(final_results)


execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
