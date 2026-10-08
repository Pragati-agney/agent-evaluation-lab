from langchain_core.tools import tool


@tool
def get_part_metadata(part_id: str) -> dict:
    """Get PLM metadata for an engineering part.

    Use this tool when information about a part's revision,
    lifecycle state, or version is required.
    """

    # Fake enterprise system for our lab.
    # Later this could be Teamcenter, a REST API,
    # database, connector, etc.
    return {
        "part_id": part_id,
        "revision": "C",
        "lifecycle_state": "Released",
        "version": "3.2",
    }