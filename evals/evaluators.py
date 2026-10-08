
def route_correct(
        expected_route:str,
        actual_route: str,
)-> bool:
    return expected_route == actual_route

def required_tools_called(
        required_tools: list[str],
        tool_calls: list[dict],
) -> bool:
    called_tools = {tool_call["name"] for tool_call in tool_calls}
    return set(required_tools) <= called_tools


def required_tool_args_present(
        required_tool_args: dict[str, dict],
        tool_calls: list[dict],
) -> bool:
    """
    For each tool, at least one of its calls must include
    the required arguments. Extra arguments are allowed.
    """

    for tool_name, required_args in required_tool_args.items():
        matched = any(
            tool_call["name"] == tool_name
            and all(
                tool_call["args"].get(key) == value
                for key, value in required_args.items()
            )
            for tool_call in tool_calls
        )

        if not matched:
            return False

    return True


def within_max_tool_calls(
        max_tool_calls: int,
        tool_calls: list[dict],
) -> bool:
    return len(tool_calls) <= max_tool_calls
