"""A fresh, deterministic tool environment for the local agent example."""

from copromem import AgentEnvironment, Tool, ToolRegistry


def make_environment(task, trial):
    del task, trial
    state = {"followed": False}

    def search_artist(artist_name: str):
        return {"artist_id": f"id-{artist_name}"}

    def follow_artist(artist_id: str):
        state["followed"] = artist_id.startswith("id-")
        return {"followed": state["followed"]}

    tools = ToolRegistry([
        Tool(name="search_artist", operation="tools.music.search_artist",
             description="Find an artist by name", access_mode="read", http_method="GET",
             parameters={"type": "object", "properties": {"artist_name": {"type": "string"}},
                         "required": ["artist_name"]}, output_slots=("artist_id",), handler=search_artist),
        Tool(name="follow_artist", operation="tools.music.follow_artist",
             description="Follow an artist", access_mode="write", http_method="POST",
             parameters={"type": "object", "properties": {"artist_id": {"type": "string"}},
                         "required": ["artist_id"]}, output_slots=("followed",), handler=follow_artist,
             succeeded=lambda result: result["followed"] is True),
    ])
    return AgentEnvironment(tools=tools, score=lambda answer: state["followed"])
