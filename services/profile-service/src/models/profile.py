from datetime import datetime
from typing import Dict, List
from pydantic import BaseModel


class LearningAbility(BaseModel):
    understanding_speed: float = 0.0
    problem_solving: float = 0.0
    critical_thinking: float = 0.0
    memory_retention: float = 0.0
    analytical_ability: float = 0.0


class LearningMotivation(BaseModel):
    intrinsic_interest: float = 0.0
    goal_oriented: float = 0.0
    extrinsic_motivation: float = 0.0


class KnowledgeCoverage(BaseModel):
    mastered: List[str] = []
    learning: List[str] = []
    not_started: List[str] = []


class InteractionStyle(BaseModel):
    visual: float = 0.0
    textual: float = 0.0
    interactive: float = 0.0
    auditory: float = 0.0


class FocusCharacteristics(BaseModel):
    avg_focus_duration_min: float = 0.0
    distraction_frequency: float = 0.0
    recommended_session_length: float = 0.0


class UserProfile(BaseModel):
    id: str
    user_id: str
    knowledge_base: Dict[str, float] = {}
    learning_ability: LearningAbility = LearningAbility()
    learning_motivation: LearningMotivation = LearningMotivation()
    knowledge_coverage: KnowledgeCoverage = KnowledgeCoverage()
    interaction_style: InteractionStyle = InteractionStyle()
    focus_characteristics: FocusCharacteristics = FocusCharacteristics()
    updated_at: datetime = datetime.now()
