from typing import Annotated, List, Optional, TypedDict

from chains import first_responder, revisor
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     ToolMessage)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from schemas import AnswerQuestion, ReviseAnswer
from tool_executor import execute_tools

MAX_ITERATIONS = 2


class State(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    draft: Optional[AnswerQuestion]
    revision: Optional[ReviseAnswer]


builder = StateGraph(State)

builder.add_node("draft", first_responder.respond)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor.respond)

builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


def event_loop(state: List[BaseMessage]) -> str:
    messages = state.get("messages", [])
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in messages)
    num_iterations = count_tool_visits
    last_message = messages[-1]

    # 1. CRITICAL FIX: Only go to 'execute_tools' if the agent actually ASKED for a tool
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        print("No need to tool call!")
        return END

    if num_iterations > MAX_ITERATIONS:
        print("Max iterations exceed!")
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
builder.set_entry_point("draft")
graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


def main():
    # prompt = (
    #     "Write about Agentic AI domain, "
    #     "List most popular latest articles."
    # )

    # out_state = graph.invoke({"messages": [HumanMessage(content=prompt)]})
    # final_messages = out_state["messages"]

    events = graph.stream(
        {
            "messages": [
                (
                    "user",
                    "Write about latest updates about aviation sector?, Compare aviation companies?, mention the financial statements of the aviation companies?",
                )
            ]
        },
        stream_mode="values",
    )
    for i, step in enumerate(events):
        print(f"Step {i}")
        step["messages"][-1].pretty_print()

        # safest generic output:
    # events = graph.stream(
    # {"messages": [("user", "Write about AI-Powered SOC / autonomous soc problem domain...")]},
    # stream_mode="updates",  # Change to "updates"
    # )

    # for i, step in enumerate(events):
    #     for node_name, update in step.items():
    #         print(f"Step {i} - Currently in node: {node_name}")
    #         # If you need the full state, you might need to merge 'update' or track it manually,
    #         # or use stream_mode="debug" for more complex inspection.
    #         print(update)


if __name__ == "__main__":
    main()
