import './App.css'
import { useEffect, useRef, useState } from 'react'

type Mode = 'beginner' | 'pro'
type Exercise = 'squat' | 'pushup' | 'lunge' | 'bicep_curl' | 'plank'

type Feedback = {
  code: string
  message: string
}

type AnalyzeResponse = {
  rep_count: number
  state: string
  is_correct_rep: boolean
  inactive: boolean
  feedback: Feedback[]
  landmarks?: { x: number; y: number }[]
  hip_knee_angle?: number
  knee_ankle_angle?: number
  shoulder_hip_angle?: number
  offset_angle?: number
  plank_hold_seconds?: number
}

type Plan = {
  total_sets: number
  reps_per_set: number
}

type PlanState = {
  current_set: number
  rep_in_set: number
  total_sets: number
  reps_per_set: number
  set_complete: boolean
  workout_complete: boolean
}

type CoachAnalyzeResponse = {
  analysis: AnalyzeResponse
  coach_message: string
  plan_state: PlanState
}

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [sessionId] = useState(() => `session-${Date.now()}`)
  const [mode, setMode] = useState<Mode>('beginner')
  const [exercise, setExercise] = useState<Exercise>('squat')
  const [plan, setPlan] = useState<Plan>({ total_sets: 3, reps_per_set: 10 })
  const [useCoach, setUseCoach] = useState(true)
  const [isRunning, setIsRunning] = useState(false)
  const [repCount, setRepCount] = useState(0)
  const [state, setState] = useState('s1')
  const [feedback, setFeedback] = useState<Feedback[]>([])
  const [lastCorrectRep, setLastCorrectRep] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [coachMessage, setCoachMessage] = useState<string | null>(null)
  const [planState, setPlanState] = useState<PlanState | null>(null)

  // Audio cues
  const correctAudioRef = useRef<HTMLAudioElement | null>(null)
  const incorrectAudioRef = useRef<HTMLAudioElement | null>(null)
  const lastSpokenRef = useRef<{ repCount: number; setIndex: number; workoutComplete: boolean }>({
    repCount: 0,
    setIndex: 0,
    workoutComplete: false,
  })

  // Skeleton pairs based on MediaPipe Pose landmark indices
  // https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#pose_landmarks
  const POSE_PAIRS: [number, number][] = [
    [11, 13],
    [13, 15], // left arm
    [12, 14],
    [14, 16], // right arm
    [11, 12], // shoulders
    [23, 24], // hips
    [11, 23],
    [12, 24], // torso
    [23, 25],
    [25, 27], // left leg
    [24, 26],
    [26, 28], // right leg
  ]

  useEffect(() => {
    if (!isRunning) {
      return
    }

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
        setError(null)
      } catch (err) {
        console.error(err)
        setError('Unable to access camera. Please allow camera permissions and refresh.')
        setIsRunning(false)
      }
    }

    startCamera()

    return () => {
      const stream = videoRef.current?.srcObject as MediaStream | null
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [isRunning])

  useEffect(() => {
    let intervalId: number | undefined

    const loop = async () => {
      if (!isRunning || !videoRef.current || !canvasRef.current) return

      const video = videoRef.current
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      canvas.width = video.videoWidth || 640
      canvas.height = video.videoHeight || 480
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      const dataUrl = canvas.toDataURL('image/jpeg', 0.5)
      const base64 = dataUrl.split(',')[1]

      try {
        const url = useCoach
          ? 'http://127.0.0.1:8000/coach/analyze'
          : 'http://127.0.0.1:8000/analyze'

        const body = useCoach
          ? {
              session_id: sessionId,
              exercise,
              mode,
              plan,
              image_b64: base64,
            }
          : {
              session_id: sessionId,
              exercise,
              mode,
              image_b64: base64,
            }

        const res = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(body),
        })

        if (!res.ok) {
          throw new Error(`Backend error: ${res.status}`)
        }

        if (useCoach) {
          const data: CoachAnalyzeResponse = await res.json()
          const { analysis, coach_message, plan_state } = data

          setRepCount(analysis.rep_count)
          setState(analysis.state)
          setFeedback(analysis.feedback)
          setCoachMessage(coach_message)
          setPlanState(plan_state)

          if (analysis.is_correct_rep && analysis.rep_count > lastCorrectRep) {
            setLastCorrectRep(analysis.rep_count)
            correctAudioRef.current?.play().catch(() => {})
          }

          const hasSevereFeedback = analysis.feedback.some((f) =>
            ['KNEES_OVER_TOES', 'DEEP_SQUAT', 'HIPS_SAG', 'PLANK_ALIGNMENT'].includes(f.code),
          )
          if (hasSevereFeedback) {
            incorrectAudioRef.current?.play().catch(() => {})
          }

          if (analysis.inactive) {
            setRepCount(analysis.rep_count)
          }

          // Speak when meaningful events happen
          maybeSpeak(coach_message, analysis, plan_state)

          // Draw skeleton overlay if landmarks are available
          if (analysis.landmarks && analysis.landmarks.length > 0) {
            const pts = analysis.landmarks.map((lm) => ({
              x: lm.x * canvas.width,
              y: lm.y * canvas.height,
            }))

            ctx.lineWidth = 3
            ctx.strokeStyle = '#22c55e'
            ctx.fillStyle = '#22c55e'

            POSE_PAIRS.forEach(([a, b]) => {
              const pa = pts[a]
              const pb = pts[b]
              if (!pa || !pb) return
              ctx.beginPath()
              ctx.moveTo(pa.x, pa.y)
              ctx.lineTo(pb.x, pb.y)
              ctx.stroke()
            })

            const jointIndices = [11, 12, 23, 24, 25, 26, 27, 28]
            jointIndices.forEach((i) => {
              const p = pts[i]
              if (!p) return
              ctx.beginPath()
              ctx.arc(p.x, p.y, 5, 0, Math.PI * 2)
              ctx.fill()
            })
          }
        } else {
          const data: AnalyzeResponse = await res.json()
          setRepCount(data.rep_count)
          setState(data.state)
          setFeedback(data.feedback)

          if (data.is_correct_rep && data.rep_count > lastCorrectRep) {
            setLastCorrectRep(data.rep_count)
            correctAudioRef.current?.play().catch(() => {})
          }

          const hasSevereFeedback = data.feedback.some((f) =>
            ['KNEES_OVER_TOES', 'DEEP_SQUAT', 'HIPS_SAG', 'PLANK_ALIGNMENT'].includes(f.code),
          )
          if (hasSevereFeedback) {
            incorrectAudioRef.current?.play().catch(() => {})
          }

          if (data.inactive) {
            setRepCount(data.rep_count)
          }

          if (data.landmarks && data.landmarks.length > 0) {
            const pts = data.landmarks.map((lm) => ({
              x: lm.x * canvas.width,
              y: lm.y * canvas.height,
            }))

            ctx.lineWidth = 3
            ctx.strokeStyle = '#22c55e'
            ctx.fillStyle = '#22c55e'

            POSE_PAIRS.forEach(([a, b]) => {
              const pa = pts[a]
              const pb = pts[b]
              if (!pa || !pb) return
              ctx.beginPath()
              ctx.moveTo(pa.x, pa.y)
              ctx.lineTo(pb.x, pb.y)
              ctx.stroke()
            })

            const jointIndices = [11, 12, 23, 24, 25, 26, 27, 28]
            jointIndices.forEach((i) => {
              const p = pts[i]
              if (!p) return
              ctx.beginPath()
              ctx.arc(p.x, p.y, 5, 0, Math.PI * 2)
              ctx.fill()
            })
          }
        }
      } catch (err) {
        console.error(err)
        setError('Error communicating with backend. Is it running on http://127.0.0.1:8000 ?')
      }
    }

    if (isRunning) {
      // Run at ~5 FPS
      intervalId = window.setInterval(loop, 200)
    }

    return () => {
      if (intervalId !== undefined) {
        window.clearInterval(intervalId)
      }
    }
  }, [isRunning, mode, sessionId, lastCorrectRep])

  const handleStart = () => {
    setRepCount(0)
    setLastCorrectRep(0)
    setFeedback([])
    setCoachMessage(null)
    setPlanState(null)
    setError(null)
    setIsRunning(true)
  }

  const handleStop = () => {
    setIsRunning(false)
  }

  const exerciseLabel = () => {
    switch (exercise) {
      case 'squat':
        return 'Squat'
      case 'pushup':
        return 'Push-up'
      case 'lunge':
        return 'Lunge'
      case 'bicep_curl':
        return 'Bicep Curl'
      case 'plank':
        return 'Plank'
      default:
        return 'Exercise'
    }
  }

  const stateLabel = () => {
    if (exercise === 'squat') {
      return state === 's1' ? 'Standing' : state === 's2' ? 'Transition' : 'Bottom'
    }
    if (exercise === 'pushup') {
      return state === 'top' ? 'Top' : state === 'bottom' ? 'Bottom' : state
    }
    if (exercise === 'lunge') {
      return state === 'stand' ? 'Stand' : state === 'lunge' ? 'Lunge' : state
    }
    if (exercise === 'bicep_curl') {
      return state === 'up' ? 'Up' : state === 'down' ? 'Down' : state
    }
    if (exercise === 'plank') {
      return state === 'on' ? 'Holding' : 'Off'
    }
    return state
  }

  const maybeSpeak = (message: string, analysis: AnalyzeResponse, ps: PlanState) => {
    if (!('speechSynthesis' in window)) return
    if (!message) return

    const last = lastSpokenRef.current

    const newSet = ps.current_set !== last.setIndex
    const newRep = analysis.is_correct_rep && analysis.rep_count > last.repCount
    const workoutJustComplete = ps.workout_complete && !last.workoutComplete

    if (!newSet && !newRep && !workoutJustComplete) return

    const utter = new SpeechSynthesisUtterance(message)
    utter.rate = 1.0
    utter.pitch = 1.0
    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(utter)

    lastSpokenRef.current = {
      repCount: analysis.rep_count,
      setIndex: ps.current_set,
      workoutComplete: ps.workout_complete,
    }
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <h1>Hackberry-Pi Trainer</h1>
        <p>Real-time form analysis using your laptop camera.</p>
      </header>

      <main className="app-main">
        <section className="video-section">
          <div className="video-container">
            <video ref={videoRef} className="video-preview" playsInline />
            <canvas ref={canvasRef} className="hidden-canvas" />
          </div>

          <div className="controls">
            <div className="mode-toggle">
              <span>Exercise:</span>
              <select
                value={exercise}
                onChange={(e) => setExercise(e.target.value as Exercise)}
                disabled={isRunning}
              >
                <option value="squat">Squat</option>
                <option value="pushup">Push-up</option>
                <option value="lunge">Lunge</option>
                <option value="bicep_curl">Bicep Curl</option>
                <option value="plank">Plank</option>
              </select>
            </div>

            <div className="mode-toggle plan-toggle">
              <span>Plan:</span>
              <label>
                Sets
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={plan.total_sets}
                  onChange={(e) =>
                    setPlan((p) => ({
                      ...p,
                      total_sets: Number(e.target.value) || 1,
                    }))
                  }
                  disabled={isRunning}
                />
              </label>
              <label>
                Reps/set
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={plan.reps_per_set}
                  onChange={(e) =>
                    setPlan((p) => ({
                      ...p,
                      reps_per_set: Number(e.target.value) || 1,
                    }))
                  }
                  disabled={isRunning}
                />
              </label>
              <label>
                LLM coach
                <input
                  type="checkbox"
                  checked={useCoach}
                  onChange={(e) => setUseCoach(e.target.checked)}
                  disabled={isRunning}
                />
              </label>
            </div>

            <div className="mode-toggle">
              <span>Mode:</span>
              <label>
                <input
                  type="radio"
                  value="beginner"
                  checked={mode === 'beginner'}
                  onChange={() => setMode('beginner')}
                  disabled={isRunning}
                />
                Beginner
              </label>
              <label>
                <input
                  type="radio"
                  value="pro"
                  checked={mode === 'pro'}
                  onChange={() => setMode('pro')}
                  disabled={isRunning}
                />
                Pro
              </label>
            </div>

            <div className="session-buttons">
              {!isRunning ? (
                <button onClick={handleStart} className="primary-btn">
                  Start Session
                </button>
              ) : (
                <button onClick={handleStop} className="secondary-btn">
                  Stop Session
                </button>
              )}
            </div>
          </div>
        </section>

        <section className="status-section">
          <div className="status-card">
            <h2>{exercise === 'plank' ? 'Hold Time (s)' : 'Reps'}</h2>
            <p className="rep-count">
              {exercise === 'plank' ? (feedback.length === 0 ? repCount : repCount) : repCount}
            </p>
          </div>

          <div className="status-card">
            <h2>State</h2>
            <p>{stateLabel()}</p>
          </div>

          <div className="status-card">
            <h2>Set / Rep</h2>
            {planState ? (
              <p>
                Set {planState.current_set}/{planState.total_sets}, rep{' '}
                {planState.rep_in_set}/{planState.reps_per_set}
              </p>
            ) : (
              <p>-</p>
            )}
          </div>

          <div className="status-card feedback-card">
            <h2>Feedback</h2>
            {useCoach ? (
              coachMessage ? (
                <p>{coachMessage}</p>
              ) : (
                <p className="feedback-ok">Coach is ready.</p>
              )
            ) : feedback.length === 0 ? (
              <p className="feedback-ok">No issues detected. Keep going!</p>
            ) : (
              <ul>
                {feedback.map((f) => (
                  <li key={f.code}>
                    <strong>{f.code}:</strong> {f.message}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {error && (
            <div className="status-card error-card">
              <h2>Error</h2>
              <p>{error}</p>
            </div>
          )}
        </section>
      </main>

      <audio ref={correctAudioRef} src="https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg" />
      <audio ref={incorrectAudioRef} src="https://actions.google.com/sounds/v1/cartoon/descending_whistle.ogg" />
    </div>
  )
}

export default App
