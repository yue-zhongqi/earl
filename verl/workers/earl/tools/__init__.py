from .base import Tool
import importlib
import os

TOOL_REGISTRY = {}

def register_tool(name: str):
    """Decorator to register a tool class"""
    def decorator(cls):
        TOOL_REGISTRY[name.lower()] = cls
        return cls
    return decorator

# Auto-import all tool modules
def _auto_import_tools():
    current_dir = os.path.dirname(__file__)
    for filename in os.listdir(current_dir):
        if filename.endswith('.py') and filename != '__init__.py' and filename != 'base.py':
            module_name = filename[:-3]
            importlib.import_module(f'.{module_name}', package=__name__)

_auto_import_tools()

def create_tool(tool_name: str, tool_config) -> Tool:
    if tool_name.lower() not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}. Available tools: {list(TOOL_REGISTRY.keys())}")
    
    tool_class = TOOL_REGISTRY[tool_name.lower()]
    return tool_class(tool_config, tool_name.lower())