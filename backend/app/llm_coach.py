import os
from typing import Dict, List

import httpx

from .models import AnalyzeResponse, PlanState

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


FORM_ISSUE_CODES: Dict[str, str] = {
    "KNEES_OVER_TOES": "keep your knees above your ankles, not far past your toes",
    "BEND_FORWARD": "hinge slightly at the hips and keep your chest proud",
    "BEND_BACKWARD": "avoid leaning too far back; keep your torso over your mid-foot",
    "HIPS_SAG": "tighten your core and keep your hips in line with shoulders and ankles",
    "PLANK_ALIGNMENT": "align your shoulders, hips, and ankles in a straight line",
    "DEEP_SQUAT": "control your depth and avoid going too deep if it feels unstable",
}


def _build_prompt(
    exercise: str,
    analysis: AnalyzeResponse,
    plan_state: PlanState,
) -> str:
    issue_codes: List[str] = [
        f.code for f in analysis.feedback if f.code in FORM_ISSUE_CODES
    ]
    unique_codes = list(dict.fromkeys(issue_codes))
    cues = [FORM_ISSUE_CODES[c] for c in unique_codes[:2]]

    status_parts: List[str] = []
    if plan_state.workout_complete:
        status_parts.append("Workout is complete.")
    elif plan_state.set_complete:
        status_parts.append("This set is complete.")
    else:
        status_parts.append(
            f"You are on rep {plan_state.rep_in_set}/{plan_state.reps_per_set} in set "
            f"{plan_state.current_set}/{plan_state.total_sets}."
        )

    if analysis.is_correct_rep:
        status_parts.append("That last rep counted as correct.")

    if analysis.inactive:
        status_parts.append("It looks like you were inactive recently or moved out of view.")

    cues_text = ""
    if cues:
        cues_text = " Form tips: " + " ".join(f"- {c}." for c in cues)

    status_line = " ".join(status_parts)

    return (
        "You are a concise, friendly fitness coach. "
        "Respond in 1–2 sentences maximum.\n\n"
        f"Exercise: {exercise}.\n"
        f"{status_line}\n"
        f"{cues_text}\n\n"
        "Give an encouraging coaching message, referencing the rep/set status and at most 1–2 concrete form cues if provided. "
        "If a set or workout is complete, congratulate the user and suggest the next step (rest or next set/workout)."
    )


async def generate_coach_message(
    exercise: str,
    analysis: AnalyzeResponse,
    plan_state: PlanState,
) -> str:
    if not ANTHROPIC_API_KEY:
        return _fallback_message(exercise, analysis, plan_state)

    prompt = _build_prompt(exercise, analysis, plan_state)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 80,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                },
            )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [])
        if content and isinstance(content, list):
            first = content[0]
            text = first.get("text") if isinstance(first, dict) else None
            if isinstance(text, str) and text.strip():
                return text.strip()
        return _fallback_message(exercise, analysis, plan_state)
    except Exception:
        return _fallback_message(exercise, analysis, plan_state)


def _fallback_message(
    exercise: str,
    analysis: AnalyzeResponse,
    plan_state: PlanState,
) -> str:
    base = (
        f"{exercise.capitalize()}s set {plan_state.current_set}/{plan_state.total_sets}, "
        f"rep {plan_state.rep_in_set}/{plan_state.reps_per_set}."
    )
    if plan_state.workout_complete:
        return base + " Great job, that’s your whole workout done. Take a break and cool down."
    if plan_state.set_complete:
        return base + " Nice work finishing this set. Take a short rest, then get ready for the next one."
    if not analysis.is_correct_rep:
        return base + " Keep your form tight and focus on smooth, controlled reps."
    return base + " Looking strong—keep the same form on the next rep."

