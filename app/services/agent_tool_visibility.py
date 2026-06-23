from sqlalchemy import or_

from app.models.agent import AgentToolCall


PUBLIC_TOOL_CALL_NAMES = {
    "create_artifact",
    "create_artifact_link",
    "create_artifact_review",
}


def public_tool_call_filter():
    return or_(
        AgentToolCall.tool_name.in_(PUBLIC_TOOL_CALL_NAMES),
        AgentToolCall.tool_name.like("write_draft:%"),
    )
