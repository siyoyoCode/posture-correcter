from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .analysis import MultiExerciseAnalyzer, compute_plan_state
from .llm_coach import generate_coach_message
from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    CoachAnalyzeRequest,
    CoachAnalyzeResponse,
)
from .pose import PoseEstimator

app = FastAPI(title="Hackberry-Pi Trainer", version="0.3.0")

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pose_estimator = PoseEstimator()
analyzer = MultiExerciseAnalyzer()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    frame = pose_estimator.decode_base64_image(req.image_b64)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    landmarks_info = pose_estimator.extract_landmarks(frame)
    if landmarks_info is None:
        # No detection; let analyzer handle inactivity in each exercise analyzer
        return analyzer.analyze(req.exercise, req.session_id, req.mode, [])

    _, coords = landmarks_info
    return analyzer.analyze(req.exercise, req.session_id, req.mode, coords)


@app.post("/coach/analyze", response_model=CoachAnalyzeResponse)
async def coach_analyze(req: CoachAnalyzeRequest) -> CoachAnalyzeResponse:
    frame = pose_estimator.decode_base64_image(req.image_b64)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    landmarks_info = pose_estimator.extract_landmarks(frame)
    if landmarks_info is None:
        analysis = analyzer.analyze(req.exercise, req.session_id, req.mode, [])
    else:
        _, coords = landmarks_info
        analysis = analyzer.analyze(req.exercise, req.session_id, req.mode, coords)

    plan_state = compute_plan_state(analysis.rep_count, req.plan)
    coach_message = await generate_coach_message(
        exercise=req.exercise.value if hasattr(req.exercise, "value") else str(req.exercise),
        analysis=analysis,
        plan_state=plan_state,
    )

    return CoachAnalyzeResponse(
        analysis=analysis,
        coach_message=coach_message,
        plan_state=plan_state,
    )


@app.get("/")
def root() -> dict:
    return {"message": "Hackberry-Pi Trainer API"}

