import os
import json
from datetime import timedelta
from langchain.agents import tool


@tool
def dummy_tool() -> str:
    """
    This is dummy tool.

    Returns:
    str: 'hello world'
    """
    print ("called the dummy tool.")
    return "hello world"
