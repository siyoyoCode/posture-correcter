import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np

from .models import AnalyzeResponse, Exercise, Feedback, LandmarkPoint, Mode, Plan, PlanState


mp_pose = mp.solutions.pose


def _angle_between(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Compute angle ABC in degrees using the dot-product formula (IJIRCST paper).
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    ba = np.array([ax - bx, ay - by], dtype=np.float32)
    bc = np.array([cx - bx, cy - by], dtype=np.float32)
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos_angle)))
    return angle


def _angle_with_vertical(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """
    Angle between line PQ and vertical line through Q.
    We approximate vertical as (q.x, q.y-1).
    """
    qx, qy = q
    vertical_ref = (qx, qy - 1.0)
    return _angle_between(p, q, vertical_ref)


@dataclass
class SessionState:
    rep_count: int = 0
    state: str = "s1"  # s1=neutral, s2=transition, s3=bottom
    state_sequence: List[str] = field(default_factory=list)
    last_movement_ts: float = field(default_factory=lambda: time.time())
    last_state_change_ts: float = field(default_factory=lambda: time.time())


class SquatAnalyzer:
    """
    Simple squat analyzer inspired by the LearnOpenCV MediaPipe trainer
    and IJIRCST FSM description. Thresholds are heuristic and can be tuned.
    """

    def __init__(self) -> None:
        # Angles in degrees for knee-vertical angle
        self.STATE_THRESH_BEGINNER = {
            "s1_max": 35.0,
            "s2_min": 35.0,
            "s2_max": 70.0,
            "s3_min": 75.0,
            "s3_max": 100.0,
        }
        # Slightly stricter for pro
        self.STATE_THRESH_PRO = {
            "s1_max": 30.0,
            "s2_min": 30.0,
            "s2_max": 65.0,
            "s3_min": 80.0,
            "s3_max": 95.0,
        }

        # Feedback thresholds (heuristic)
        self.FEEDBACK_THRESH = {
            "bend_forward_min": 10.0,
            "bend_forward_max": 20.0,
            "bend_backward_min": 40.0,
            "bend_backward_max": 60.0,
            "knees_over_toes_min": 25.0,
            "deep_squat_min": 95.0,
        }

        # Seconds
        self.INACTIVE_THRESH = 15.0

        self.sessions: Dict[str, SessionState] = {}

    def _get_state_thresholds(self, mode: Mode) -> Dict[str, float]:
        if mode == Mode.pro:
            return self.STATE_THRESH_PRO
        return self.STATE_THRESH_BEGINNER

    def _get_or_create_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState()
        return self.sessions[session_id]

    def _classify_state(self, knee_vertical_angle: float, mode: Mode) -> str:
        t = self._get_state_thresholds(mode)
        if knee_vertical_angle <= t["s1_max"]:
            return "s1"
        if t["s2_min"] < knee_vertical_angle <= t["s2_max"]:
            return "s2"
        if t["s3_min"] <= knee_vertical_angle <= t["s3_max"]:
            return "s3"
        # Outside ranges, treat as transition
        return "s2"

    def _update_fsm_and_reps(
        self, session: SessionState, new_state: str
    ) -> bool:
        """
        Update state machine and possibly increment rep.
        Returns True if this transition completed a correct rep.
        """
        is_correct_rep = False
        now = time.time()

        if new_state != session.state:
            session.last_state_change_ts = now
            session.last_movement_ts = now

            # Track sequence only while not in s1
            if new_state != "s1":
                session.state_sequence.append(new_state)
                # Cap length to avoid unbounded growth
                if len(session.state_sequence) > 5:
                    session.state_sequence.pop(0)
            else:
                # We just returned to neutral; check if we saw s2 and s3
                if "s2" in session.state_sequence and "s3" in session.state_sequence:
                    session.rep_count += 1
                    is_correct_rep = True
                # Reset for next cycle
                session.state_sequence.clear()

            session.state = new_state
        else:
            # Same state; still consider this movement
            session.last_movement_ts = now

        return is_correct_rep

    def _check_inactive(self, session: SessionState) -> bool:
        now = time.time()
        if now - session.last_movement_ts > self.INACTIVE_THRESH:
            # Reset session counters but keep same object
            session.rep_count = 0
            session.state_sequence.clear()
            session.state = "s1"
            session.last_movement_ts = now
            session.last_state_change_ts = now
            return True
        return False

    def analyze_landmarks(
        self,
        session_id: str,
        mode: Mode,
        landmarks: List[Tuple[float, float]],
    ) -> AnalyzeResponse:
        """
        Analyze a single frame's landmarks and return squat analysis.
        """
        session = self._get_or_create_session(session_id)

        # Short-circuit if landmarks missing critical points
        try:
            lmk = mp_pose.PoseLandmark
            hip = landmarks[lmk.LEFT_HIP]
            knee = landmarks[lmk.LEFT_KNEE]
            ankle = landmarks[lmk.LEFT_ANKLE]
            shoulder = landmarks[lmk.LEFT_SHOULDER]
            nose = landmarks[lmk.NOSE]
        except Exception:
            inactive = self._check_inactive(session)
            return AnalyzeResponse(
                rep_count=session.rep_count,
                state=session.state,
                is_correct_rep=False,
                inactive=inactive,
                feedback=[Feedback(code="NO_DETECTION", message="Person not fully visible")],
            )

        # Compute angles
        hip_knee_angle = _angle_with_vertical(hip, knee)
        knee_ankle_angle = _angle_with_vertical(knee, ankle)
        shoulder_hip_angle = _angle_with_vertical(shoulder, hip)
        offset_angle = _angle_between(shoulder, nose, (shoulder[0] + 1.0, shoulder[1]))

        # Classify state and update FSM
        state = self._classify_state(hip_knee_angle, mode)
        is_correct_rep = self._update_fsm_and_reps(session, state)

        # Inactivity
        inactive = self._check_inactive(session)

        # Feedback
        feedback: List[Feedback] = []

        # Offset/front-view warning
        if offset_angle > 25.0:
            feedback.append(
                Feedback(
                    code="OFFSET_VIEW",
                    message="Turn more sideways to the camera for better squat analysis.",
                )
            )

        # Bend forward/backward at hips
        if shoulder_hip_angle < self.FEEDBACK_THRESH["bend_forward_min"]:
            feedback.append(
                Feedback(
                    code="BEND_FORWARD",
                    message="Bend slightly forward at the hips.",
                )
            )
        elif shoulder_hip_angle > self.FEEDBACK_THRESH["bend_backward_max"]:
            feedback.append(
                Feedback(
                    code="BEND_BACKWARD",
                    message="Avoid leaning too far back.",
                )
            )

        # Depth guidance
        if state == "s2" and hip_knee_angle < self._get_state_thresholds(mode)["s2_min"] + 5:
            feedback.append(
                Feedback(
                    code="GO_LOWER",
                    message="Lower your hips more to reach squat depth.",
                )
            )

        # Deep squat (potentially unsafe depending on context)
        if hip_knee_angle > self.FEEDBACK_THRESH["deep_squat_min"]:
            feedback.append(
                Feedback(
                    code="DEEP_SQUAT",
                    message="You are going very deep; ensure this matches your training goal.",
                )
            )

        # Knees over toes
        if knee_ankle_angle > self.FEEDBACK_THRESH["knees_over_toes_min"]:
            feedback.append(
                Feedback(
                    code="KNEES_OVER_TOES",
                    message="Keep your knees from moving too far past your toes.",
                )
            )

        return AnalyzeResponse(
            rep_count=session.rep_count,
            state=session.state,
            is_correct_rep=is_correct_rep,
            inactive=inactive,
            feedback=feedback,
            hip_knee_angle=hip_knee_angle,
            knee_ankle_angle=knee_ankle_angle,
            shoulder_hip_angle=shoulder_hip_angle,
            offset_angle=offset_angle,
            landmarks=[LandmarkPoint(x=float(x), y=float(y)) for (x, y) in landmarks],
        )


class PushupAnalyzer:
    """
    Side-view push-up analyzer using elbow angle and body line.
    """

    def __init__(self) -> None:
        self.sessions: Dict[str, SessionState] = {}
        # Elbow angle thresholds (degrees)
        self.TOP_MAX = 25.0   # near-locked out
        self.BOTTOM_MIN = 80.0  # deep enough
        self.INACTIVE_THRESH = 15.0

    def _get_or_create_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(state="top")
        return self.sessions[session_id]

    def _check_inactive(self, session: SessionState) -> bool:
        now = time.time()
        if now - session.last_movement_ts > self.INACTIVE_THRESH:
            session.rep_count = 0
            session.state = "top"
            session.state_sequence.clear()
            session.last_movement_ts = now
            session.last_state_change_ts = now
            return True
        return False

    def analyze(
        self, session_id: str, mode: Mode, landmarks: List[Tuple[float, float]]
    ) -> AnalyzeResponse:
        session = self._get_or_create_session(session_id)

        try:
            lmk = mp_pose.PoseLandmark
            shoulder = landmarks[lmk.LEFT_SHOULDER]
            elbow = landmarks[lmk.LEFT_ELBOW]
            wrist = landmarks[lmk.LEFT_WRIST]
            hip = landmarks[lmk.LEFT_HIP]
            ankle = landmarks[lmk.LEFT_ANKLE]
        except Exception:
            inactive = self._check_inactive(session)
            return AnalyzeResponse(
                rep_count=session.rep_count,
                state=session.state,
                is_correct_rep=False,
                inactive=inactive,
                feedback=[Feedback(code="NO_DETECTION", message="Person not fully visible")],
            )

        elbow_angle = _angle_between(shoulder, elbow, wrist)
        body_straight_angle = _angle_between(shoulder, hip, ankle)

        feedback: List[Feedback] = []

        # Body line: encourage roughly straight plank (180 degrees)
        if body_straight_angle < 160.0:
            feedback.append(
                Feedback(
                    code="HIPS_SAG",
                    message="Keep your body straighter; avoid sagging hips.",
                )
            )

        # FSM: top -> bottom -> top
        now = time.time()
        session.last_movement_ts = now

        is_correct_rep = False
        if session.state == "top":
            if elbow_angle > self.BOTTOM_MIN:
                session.state = "bottom"
                session.last_state_change_ts = now
        elif session.state == "bottom":
            if elbow_angle < self.TOP_MAX:
                session.state = "top"
                session.rep_count += 1
                is_correct_rep = True
                session.last_state_change_ts = now

        inactive = self._check_inactive(session)

        return AnalyzeResponse(
            rep_count=session.rep_count,
            state=session.state,
            is_correct_rep=is_correct_rep,
            inactive=inactive,
            feedback=feedback,
            hip_knee_angle=None,
            knee_ankle_angle=None,
            shoulder_hip_angle=None,
            offset_angle=None,
            landmarks=[LandmarkPoint(x=float(x), y=float(y)) for (x, y) in landmarks],
        )


class LungeAnalyzer:
    """
    Simple forward lunge analyzer based on front leg knee angle and depth.
    Assumes left leg is the front leg in side view.
    """

    def __init__(self) -> None:
        self.sessions: Dict[str, SessionState] = {}
        self.INACTIVE_THRESH = 15.0

    def _get_or_create_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(state="stand")
        return self.sessions[session_id]

    def _check_inactive(self, session: SessionState) -> bool:
        now = time.time()
        if now - session.last_movement_ts > self.INACTIVE_THRESH:
            session.rep_count = 0
            session.state = "stand"
            session.state_sequence.clear()
            session.last_movement_ts = now
            session.last_state_change_ts = now
            return True
        return False

    def analyze(
        self, session_id: str, mode: Mode, landmarks: List[Tuple[float, float]]
    ) -> AnalyzeResponse:
        session = self._get_or_create_session(session_id)

        try:
            lmk = mp_pose.PoseLandmark
            hip = landmarks[lmk.LEFT_HIP]
            knee = landmarks[lmk.LEFT_KNEE]
            ankle = landmarks[lmk.LEFT_ANKLE]
        except Exception:
            inactive = self._check_inactive(session)
            return AnalyzeResponse(
                rep_count=session.rep_count,
                state=session.state,
                is_correct_rep=False,
                inactive=inactive,
                feedback=[Feedback(code="NO_DETECTION", message="Person not fully visible")],
            )

        knee_vertical_angle = _angle_with_vertical(hip, knee)
        knee_ankle_angle = _angle_with_vertical(knee, ankle)

        feedback: List[Feedback] = []

        # Depth guidance
        if knee_vertical_angle < 40.0:
            feedback.append(
                Feedback(
                    code="LUNGE_NOT_DEEP",
                    message="Step further and lower your back knee for a deeper lunge.",
                )
            )
        # Knees over toes
        if knee_ankle_angle > 25.0:
            feedback.append(
                Feedback(
                    code="KNEES_OVER_TOES",
                    message="Keep your front knee above your ankle, not far past your toes.",
                )
            )

        # FSM: stand -> lunge -> stand
        now = time.time()
        session.last_movement_ts = now
        is_correct_rep = False

        if session.state == "stand":
            if knee_vertical_angle > 60.0:
                session.state = "lunge"
                session.last_state_change_ts = now
        elif session.state == "lunge":
            if knee_vertical_angle < 40.0:
                session.state = "stand"
                session.rep_count += 1
                is_correct_rep = True
                session.last_state_change_ts = now

        inactive = self._check_inactive(session)

        return AnalyzeResponse(
            rep_count=session.rep_count,
            state=session.state,
            is_correct_rep=is_correct_rep,
            inactive=inactive,
            feedback=feedback,
            hip_knee_angle=knee_vertical_angle,
            knee_ankle_angle=knee_ankle_angle,
            shoulder_hip_angle=None,
            offset_angle=None,
            landmarks=[LandmarkPoint(x=float(x), y=float(y)) for (x, y) in landmarks],
        )


class BicepCurlAnalyzer:
    """
    Single-arm bicep curl based on elbow flexion/extension.
    Uses right arm by default.
    """

    def __init__(self) -> None:
        self.sessions: Dict[str, SessionState] = {}
        self.INACTIVE_THRESH = 15.0
        self.TOP_MIN = 45.0   # fully curled (smaller angle)
        self.BOTTOM_MAX = 160.0  # arm mostly straight

    def _get_or_create_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(state="down")
        return self.sessions[session_id]

    def _check_inactive(self, session: SessionState) -> bool:
        now = time.time()
        if now - session.last_movement_ts > self.INACTIVE_THRESH:
            session.rep_count = 0
            session.state = "down"
            session.state_sequence.clear()
            session.last_movement_ts = now
            session.last_state_change_ts = now
            return True
        return False

    def analyze(
        self, session_id: str, mode: Mode, landmarks: List[Tuple[float, float]]
    ) -> AnalyzeResponse:
        session = self._get_or_create_session(session_id)

        try:
            lmk = mp_pose.PoseLandmark
            shoulder = landmarks[lmk.RIGHT_SHOULDER]
            elbow = landmarks[lmk.RIGHT_ELBOW]
            wrist = landmarks[lmk.RIGHT_WRIST]
        except Exception:
            inactive = self._check_inactive(session)
            return AnalyzeResponse(
                rep_count=session.rep_count,
                state=session.state,
                is_correct_rep=False,
                inactive=inactive,
                feedback=[Feedback(code="NO_DETECTION", message="Arm not clearly visible")],
            )

        elbow_angle = _angle_between(shoulder, elbow, wrist)

        feedback: List[Feedback] = []

        now = time.time()
        session.last_movement_ts = now
        is_correct_rep = False

        if session.state == "down":
            if elbow_angle < self.TOP_MIN:
                session.state = "up"
                session.last_state_change_ts = now
        elif session.state == "up":
            if elbow_angle > self.BOTTOM_MAX:
                session.state = "down"
                session.rep_count += 1
                is_correct_rep = True
                session.last_state_change_ts = now

        inactive = self._check_inactive(session)

        return AnalyzeResponse(
            rep_count=session.rep_count,
            state=session.state,
            is_correct_rep=is_correct_rep,
            inactive=inactive,
            feedback=feedback,
            hip_knee_angle=None,
            knee_ankle_angle=None,
            shoulder_hip_angle=None,
            offset_angle=None,
            landmarks=[LandmarkPoint(x=float(x), y=float(y)) for (x, y) in landmarks],
        )


class PlankAnalyzer:
    """
    Plank analyzer: tracks how long user maintains roughly straight body line.
    """

    def __init__(self) -> None:
        self.sessions: Dict[str, SessionState] = {}
        self.INACTIVE_THRESH = 10.0
        self.MAX_DEVIATION = 25.0  # degrees from straight line

    def _get_or_create_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(state="off")
        return self.sessions[session_id]

    def _check_inactive(self, session: SessionState) -> bool:
        now = time.time()
        if now - session.last_movement_ts > self.INACTIVE_THRESH:
            session.rep_count = 0
            session.state = "off"
            session.state_sequence.clear()
            session.last_movement_ts = now
            session.last_state_change_ts = now
            return True
        return False

    def analyze(
        self, session_id: str, mode: Mode, landmarks: List[Tuple[float, float]]
    ) -> AnalyzeResponse:
        session = self._get_or_create_session(session_id)

        try:
            lmk = mp_pose.PoseLandmark
            shoulder = landmarks[lmk.LEFT_SHOULDER]
            hip = landmarks[lmk.LEFT_HIP]
            ankle = landmarks[lmk.LEFT_ANKLE]
        except Exception:
            inactive = self._check_inactive(session)
            return AnalyzeResponse(
                rep_count=session.rep_count,
                state=session.state,
                is_correct_rep=False,
                inactive=inactive,
                feedback=[Feedback(code="NO_DETECTION", message="Person not fully visible for plank")],
                plank_hold_seconds=None,
            )

        body_angle = _angle_between(shoulder, hip, ankle)
        deviation = abs(180.0 - body_angle)

        feedback: List[Feedback] = []
        now = time.time()
        session.last_movement_ts = now

        if deviation > self.MAX_DEVIATION:
            feedback.append(
                Feedback(
                    code="PLANK_ALIGNMENT",
                    message="Keep your shoulders, hips and ankles in a straight line.",
                )
            )

        # Use state "on"/"off" and rep_count as seconds (approximate)
        if deviation <= self.MAX_DEVIATION:
            if session.state != "on":
                session.state = "on"
                session.last_state_change_ts = now
            # Update hold time in seconds
            hold_seconds = now - session.last_state_change_ts
        else:
            session.state = "off"
            hold_seconds = 0.0
            session.last_state_change_ts = now

        inactive = self._check_inactive(session)

        return AnalyzeResponse(
            rep_count=int(hold_seconds),
            state=session.state,
            is_correct_rep=False,
            inactive=inactive,
            feedback=feedback,
            hip_knee_angle=None,
            knee_ankle_angle=None,
            shoulder_hip_angle=body_angle,
            offset_angle=None,
            landmarks=[LandmarkPoint(x=float(x), y=float(y)) for (x, y) in landmarks],
            plank_hold_seconds=hold_seconds,
        )


class MultiExerciseAnalyzer:
    """
    High-level dispatcher for multiple exercises.
    """

    def __init__(self) -> None:
        self.squat = SquatAnalyzer()
        self.pushup = PushupAnalyzer()
        self.lunge = LungeAnalyzer()
        self.curl = BicepCurlAnalyzer()
        self.plank = PlankAnalyzer()

    def analyze(
        self,
        exercise: Exercise,
        session_id: str,
        mode: Mode,
        landmarks: List[Tuple[float, float]],
    ) -> AnalyzeResponse:
        if exercise == Exercise.squat:
            return self.squat.analyze_landmarks(session_id, mode, landmarks)
        if exercise == Exercise.pushup:
            return self.pushup.analyze(session_id, mode, landmarks)
        if exercise == Exercise.lunge:
            return self.lunge.analyze(session_id, mode, landmarks)
        if exercise == Exercise.bicep_curl:
            return self.curl.analyze(session_id, mode, landmarks)
        if exercise == Exercise.plank:
            return self.plank.analyze(session_id, mode, landmarks)

        # Fallback to squat
        return self.squat.analyze_landmarks(session_id, mode, landmarks)


def compute_plan_state(rep_count: int, plan: Plan) -> PlanState:
    """
    Derive per-set position and completion flags from total rep count and plan.
    """
    total_reps = max(plan.total_sets * plan.reps_per_set, 1)
    capped_reps = min(max(rep_count, 0), total_reps)

    if capped_reps == 0:
        current_set = 1
        rep_in_set = 0
    else:
        current_set = (capped_reps - 1) // plan.reps_per_set + 1
        rep_in_set = (capped_reps - 1) % plan.reps_per_set + 1

    set_complete = rep_in_set == plan.reps_per_set and capped_reps > 0
    workout_complete = capped_reps >= total_reps

    return PlanState(
        current_set=current_set,
        rep_in_set=rep_in_set,
        total_sets=plan.total_sets,
        reps_per_set=plan.reps_per_set,
        set_complete=set_complete,
        workout_complete=workout_complete,
    )

