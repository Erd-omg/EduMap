from typing import Any, Dict, List, Optional, TypedDict


class EduMapState(TypedDict, total=False):
    """LangGraph state type for EduMap orchestration."""
    task_input: str
    task_type: str
    current_phase: str
    agent_results: Dict[str, Any]
    errors: List[Dict[str, str]]
    audit_log: List[Dict[str, Any]]
    user_id: str
    session_id: str
    knowledge_point_id: Optional[str]
