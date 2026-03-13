from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class Mode(str, Enum):
    beginner = "beginner"
    pro = "pro"


class Exercise(str, Enum):
    squat = "squat"
    pushup = "pushup"
    lunge = "lunge"
    bicep_curl = "bicep_curl"
    plank = "plank"


class AnalyzeRequest(BaseModel):
    session_id: str
    exercise: Exercise = Exercise.squat
    mode: Mode = Mode.beginner
    # Raw base64 string WITHOUT data: prefix
    image_b64: str


class Feedback(BaseModel):
    code: str
    message: str


class LandmarkPoint(BaseModel):
    x: float  # normalized [0,1]
    y: float  # normalized [0,1]


class AnalyzeResponse(BaseModel):
    rep_count: int
    state: str
    is_correct_rep: bool
    inactive: bool
    feedback: List[Feedback]
    # Optional debug angles for tuning on the client
    hip_knee_angle: Optional[float] = None
    knee_ankle_angle: Optional[float] = None
    shoulder_hip_angle: Optional[float] = None
    offset_angle: Optional[float] = None
    # Optional landmarks for client-side skeleton overlay
    landmarks: Optional[List[LandmarkPoint]] = None
    # Optional: plank hold duration (seconds) when exercise == plank
    plank_hold_seconds: Optional[float] = None


class Plan(BaseModel):
    total_sets: int
    reps_per_set: int
    current_set: Optional[int] = None
    done_reps: Optional[int] = None


class PlanState(BaseModel):
    current_set: int
    rep_in_set: int
    total_sets: int
    reps_per_set: int
    set_complete: bool
    workout_complete: bool


class CoachAnalyzeRequest(BaseModel):
    session_id: str
    exercise: Exercise
    mode: Mode = Mode.beginner
    plan: Plan
    image_b64: str


class CoachAnalyzeResponse(BaseModel):
    analysis: AnalyzeResponse
    coach_message: str
    plan_state: PlanState

