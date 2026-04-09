import { useEffect, useRef, useState, useCallback } from 'react'

// ── MediaPipe globals injected via CDN ──────────────────────────────────────
declare global {
  interface Window {
    Hands: new (config: object) => HandsInstance
    Camera: new (video: HTMLVideoElement, config: object) => CameraInstance
    drawConnectors: (ctx: CanvasRenderingContext2D, lm: Landmark[], conns: unknown, style: object) => void
    drawLandmarks: (ctx: CanvasRenderingContext2D, lm: Landmark[], style: object) => void
    HAND_CONNECTIONS: unknown
  }
}
interface Landmark { x: number; y: number; z: number }
interface HandsInstance {
  setOptions(o: object): void
  onResults(fn: (r: HandsResult) => void): void
  send(input: { image: HTMLVideoElement }): Promise<void>
}
interface CameraInstance { start(): Promise<void>; stop(): void }
interface HandsResult { multiHandLandmarks?: Landmark[][] }

const CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe'
const SCRIPTS = [
  `${CDN}/camera_utils/camera_utils.js`,
  `${CDN}/drawing_utils/drawing_utils.js`,
  `${CDN}/hands/hands.js`,
]

const DEBOUNCE_MS = 800
const SWIPE_THRESHOLD = 0.04
const SWIPE_BUF = 10

function loadScript(src: string): Promise<void> {
  return new Promise((res, rej) => {
    if (document.querySelector(`script[src="${src}"]`)) { res(); return }
    const s = document.createElement('script')
    s.src = src; s.crossOrigin = 'anonymous'
    s.onload = () => res(); s.onerror = rej
    document.head.appendChild(s)
  })
}

export default function GestureController() {
  const [enabled, setEnabled] = useState(false)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState('Click ON to start camera')
  const [gestureName, setGestureName] = useState('—')
  const [gestureActive, setGestureActive] = useState(false)
  const [error, setError] = useState('')

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const cursorRef = useRef<HTMLDivElement>(null)

  const handsRef = useRef<HandsInstance | null>(null)
  const cameraRef = useRef<CameraInstance | null>(null)
  const rafRef = useRef<number | null>(null)

  const cursorX = useRef(window.innerWidth / 2)
  const cursorY = useRef(window.innerHeight / 2)
  const targetX = useRef(window.innerWidth / 2)
  const targetY = useRef(window.innerHeight / 2)
  const frozen = useRef(false)
  const lastHovered = useRef<Element | null>(null)
  const lastFired = useRef<Record<string, number>>({})
  const swipeBuf = useRef<number[]>([])

  // ── Gesture recognition ──────────────────────────────────
  const recognise = useCallback((lm: Landmark[]): string => {
    const thumbExt  = lm[4].y  < lm[3].y
    const indexExt  = lm[8].y  < lm[6].y
    const middleExt = lm[12].y < lm[10].y
    const ringExt   = lm[16].y < lm[14].y
    const pinkyExt  = lm[20].y < lm[18].y
    const extCount  = [thumbExt, indexExt, middleExt, ringExt, pinkyExt].filter(Boolean).length

    const pinchDist = Math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
    if (pinchDist < 0.06) return 'PINCH'

    if (indexExt && middleExt && ringExt && !thumbExt && !pinkyExt) return 'THREE_FINGERS'
    if (indexExt && middleExt && !ringExt && !pinkyExt) return 'TWO_FINGERS'
    if (extCount === 5) return 'OPEN_PALM'
    if (!indexExt && !middleExt && !ringExt && !pinkyExt) return 'FIST'

    if (indexExt && !middleExt && !ringExt && !pinkyExt) {
      const buf = swipeBuf.current
      buf.push(lm[8].x)
      if (buf.length > SWIPE_BUF) buf.shift()
      if (buf.length === SWIPE_BUF) {
        const vel = buf[SWIPE_BUF - 1] - buf[0]
        if (vel > SWIPE_THRESHOLD) return 'SWIPE_RIGHT'
        if (vel < -SWIPE_THRESHOLD) return 'SWIPE_LEFT'
      }
      return 'POINT'
    }
    return 'NONE'
  }, [])

  const canFire = useCallback((g: string) => {
    if (g === 'POINT') return true
    const now = Date.now()
    if (now - (lastFired.current[g] ?? 0) < DEBOUNCE_MS) return false
    lastFired.current[g] = now
    return true
  }, [])

  function spawnRipple(x: number, y: number) {
    const r = document.createElement('div')
    r.style.cssText = `position:fixed;pointer-events:none;z-index:99998;width:36px;height:36px;border-radius:50%;border:2px solid #22d3ee;left:${x}px;top:${y}px;transform:translate(-50%,-50%) scale(0);animation:gesture-ripple .4s ease-out forwards;`
    document.body.appendChild(r)
    setTimeout(() => r.remove(), 450)
  }

  const dispatch = useCallback((gesture: string, lm: Landmark[]) => {
    if (!canFire(gesture)) return
    switch (gesture) {
      case 'POINT':
        if (!frozen.current) {
          targetX.current = lm[8].x * window.innerWidth
          targetY.current = lm[8].y * window.innerHeight
        }
        break
      case 'PINCH': {
        targetX.current = lm[8].x * window.innerWidth
        targetY.current = lm[8].y * window.innerHeight
        const el = document.elementFromPoint(cursorX.current, cursorY.current)
        if (el) {
          el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
          spawnRipple(cursorX.current, cursorY.current)
        }
        break
      }
      case 'OPEN_PALM':
        frozen.current = !frozen.current
        break
      case 'SWIPE_LEFT':
        swipeBuf.current.length = 0
        history.back()
        break
      case 'SWIPE_RIGHT':
        swipeBuf.current.length = 0
        history.forward()
        break
      case 'TWO_FINGERS':
        window.scrollBy({ top: -140, behavior: 'smooth' })
        break
      case 'FIST':
        window.scrollBy({ top: 140, behavior: 'smooth' })
        break
      case 'THREE_FINGERS':
        setEnabled(false)
        break
    }
  }, [canFire])

  // ── Cursor animation loop ────────────────────────────────
  const startCursorLoop = useCallback(() => {
    const loop = () => {
      rafRef.current = requestAnimationFrame(loop)
      if (!frozen.current) {
        cursorX.current += (targetX.current - cursorX.current) * 0.2
        cursorY.current += (targetY.current - cursorY.current) * 0.2
      }
      const el = cursorRef.current
      if (el) { el.style.left = cursorX.current + 'px'; el.style.top = cursorY.current + 'px' }

      const hovered = document.elementFromPoint(cursorX.current, cursorY.current)
      if (hovered !== lastHovered.current) {
        lastHovered.current?.classList.remove('gesture-hover')
        if (hovered && hovered !== document.body && hovered !== document.documentElement) {
          hovered.classList.add('gesture-hover')
        }
        lastHovered.current = hovered
      }
    }
    loop()
  }, [])

  // ── MediaPipe result handler ─────────────────────────────
  const onResults = useCallback((results: HandsResult) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const container = canvas.parentElement!
    canvas.width  = container.offsetWidth
    canvas.height = container.offsetHeight
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (!results.multiHandLandmarks?.length) {
      setGestureName('—'); setGestureActive(false); return
    }
    const lm = results.multiHandLandmarks[0]
    const mirrored = lm.map(p => ({ x: 1 - p.x, y: p.y, z: p.z }))

    if (window.drawConnectors && window.HAND_CONNECTIONS) {
      window.drawConnectors(ctx, mirrored, window.HAND_CONNECTIONS, { color: 'rgba(99,102,241,0.8)', lineWidth: 1.5 })
      window.drawLandmarks(ctx, mirrored, { color: 'rgba(34,211,238,0.9)', lineWidth: 1, radius: 2 })
    }

    const gesture = recognise(mirrored)
    setGestureName(gesture === 'NONE' ? '—' : gesture.replace('_', ' '))
    setGestureActive(gesture !== 'NONE')
    dispatch(gesture, mirrored)
  }, [recognise, dispatch])

  // ── Start / stop ─────────────────────────────────────────
  useEffect(() => {
    if (!enabled) {
      cameraRef.current?.stop(); cameraRef.current = null
      handsRef.current = null
      if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
      lastHovered.current?.classList.remove('gesture-hover'); lastHovered.current = null
      frozen.current = false
      setGestureName('—'); setGestureActive(false)
      setStatus('Click ON to start camera')
      return
    }

    let cancelled = false
    async function start() {
      setLoading(true); setError('')
      try {
        setStatus('Loading MediaPipe…')
        await Promise.all(SCRIPTS.map(loadScript))
        if (cancelled) return

        setStatus('Starting camera…')
        const video = videoRef.current!
        const hands = new window.Hands({ locateFile: (f: string) => `${CDN}/hands/${f}` })
        hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.7, minTrackingConfidence: 0.6 })
        hands.onResults(onResults)
        handsRef.current = hands

        const cam = new window.Camera(video, {
          onFrame: async () => { if (handsRef.current) await handsRef.current.send({ image: video }) },
          width: 320, height: 240,
        })
        await cam.start()
        cameraRef.current = cam
        if (cancelled) { cam.stop(); return }

        startCursorLoop()
        setStatus('Tracking…')
      } catch (err: unknown) {
        if (!cancelled) {
          const msg = err instanceof Error ? err.message : String(err)
          setError(msg)
          setEnabled(false)
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    start()
    return () => { cancelled = true }
  }, [enabled, onResults, startCursorLoop])

  // ── Inject global CSS once ────────────────────────────────
  useEffect(() => {
    const id = 'gesture-global-css'
    if (document.getElementById(id)) return
    const style = document.createElement('style')
    style.id = id
    style.textContent = `
      .gesture-hover { outline: 2px solid rgba(99,102,241,0.7) !important; outline-offset: 2px; }
      @keyframes gesture-ripple { to { transform: translate(-50%,-50%) scale(2.5); opacity: 0; } }
    `
    document.head.appendChild(style)
  }, [])

  return (
    <>
      {/* Virtual cursor */}
      <div
        ref={cursorRef}
        style={{
          position: 'fixed', top: 0, left: 0, pointerEvents: 'none', zIndex: 99999,
          transform: 'translate(-50%,-50%)', opacity: enabled ? 1 : 0, transition: 'opacity .2s',
        }}
      >
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
          <circle cx="14" cy="14" r="5" fill="white" opacity="0.95"/>
          <circle cx="14" cy="14" r="8" stroke="rgba(99,102,241,1)" strokeWidth="1.5" opacity="0.7"/>
          <circle cx="14" cy="14" r="12" stroke="rgba(34,211,238,0.5)" strokeWidth="1"/>
        </svg>
      </div>

      {/* Panel */}
      <DraggablePanel
        enabled={enabled}
        loading={loading}
        status={status}
        gestureName={gestureName}
        gestureActive={gestureActive}
        error={error}
        onToggle={() => setEnabled(v => !v)}
        videoRef={videoRef}
        canvasRef={canvasRef}
      />
    </>
  )
}

// ── Draggable panel ───────────────────────────────────────
interface PanelProps {
  enabled: boolean; loading: boolean; status: string
  gestureName: string; gestureActive: boolean; error: string
  onToggle(): void
  videoRef: React.RefObject<HTMLVideoElement>
  canvasRef: React.RefObject<HTMLCanvasElement>
}

function DraggablePanel({ enabled, loading, status, gestureName, gestureActive, error, onToggle, videoRef, canvasRef }: PanelProps) {
  const panelRef = useRef<HTMLDivElement>(null)
  const drag = useRef<{ ox: number; oy: number } | null>(null)

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!drag.current || !panelRef.current) return
      panelRef.current.style.right  = 'auto'
      panelRef.current.style.bottom = 'auto'
      panelRef.current.style.left   = e.clientX - drag.current.ox + 'px'
      panelRef.current.style.top    = e.clientY - drag.current.oy + 'px'
    }
    const onUp = () => { drag.current = null; document.body.style.userSelect = '' }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
  }, [])

  const onDragStart = (e: React.MouseEvent) => {
    if (!panelRef.current) return
    const r = panelRef.current.getBoundingClientRect()
    drag.current = { ox: e.clientX - r.left, oy: e.clientY - r.top }
    document.body.style.userSelect = 'none'
  }

  return (
    <div
      ref={panelRef}
      style={{
        position: 'fixed', bottom: '1.5rem', right: '1.5rem', zIndex: 99990,
        width: 210, background: 'rgba(10,12,20,0.88)', backdropFilter: 'blur(20px) saturate(180%)',
        border: '1px solid rgba(255,255,255,0.12)', borderRadius: '1.25rem',
        padding: '1rem', boxShadow: '0 8px 40px rgba(0,0,0,.5)', userSelect: 'none',
      }}
    >
      {/* header */}
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'.75rem', cursor:'grab' }} onMouseDown={onDragStart}>
        <span style={{ fontSize:'.7rem', fontWeight:700, letterSpacing:'.06em', textTransform:'uppercase', color:'#94a3b8' }}>
          Gesture Control
        </span>
        <button
          onClick={onToggle}
          disabled={loading}
          style={{
            fontSize:'.7rem', padding:'.2rem .55rem', borderRadius:999, cursor:'pointer',
            border: enabled ? '1px solid #6366f1' : '1px solid rgba(255,255,255,0.1)',
            background: enabled ? 'rgba(99,102,241,0.3)' : 'rgba(255,255,255,0.05)',
            color: enabled ? '#a5b4fc' : '#e2e8f0', fontWeight:600,
          }}
        >
          {loading ? '…' : enabled ? 'ON' : 'OFF'}
        </button>
      </div>

      {/* camera feed */}
      <div style={{ position:'relative', width:'100%', aspectRatio:'4/3', borderRadius:'.75rem', overflow:'hidden', background:'#000', marginBottom:'.75rem' }}>
        <video ref={videoRef} playsInline style={{ position:'absolute', inset:0, width:'100%', height:'100%', objectFit:'cover', transform:'scaleX(-1)', opacity:0 }} />
        <canvas ref={canvasRef} style={{ position:'absolute', inset:0, width:'100%', height:'100%' }} />
        {!enabled && (
          <div style={{ position:'absolute', inset:0, display:'flex', alignItems:'center', justifyContent:'center', fontSize:'.7rem', color:'#64748b' }}>
            camera off
          </div>
        )}
      </div>

      {/* gesture label */}
      <div style={{ fontSize:'.7rem', padding:'.3rem .7rem', borderRadius:'.5rem', background:'rgba(255,255,255,.05)', border:'1px solid rgba(255,255,255,0.1)', display:'flex', alignItems:'center', gap:'.4rem' }}>
        <div style={{ width:7, height:7, borderRadius:'50%', flexShrink:0, background: gestureActive ? '#22d3ee' : '#374151', boxShadow: gestureActive ? '0 0 6px #22d3ee' : 'none', transition:'all .2s' }} />
        <span style={{ flex:1, color:'#e2e8f0', fontWeight:600 }}>{gestureName}</span>
      </div>

      {error
        ? <div style={{ fontSize:'.7rem', color:'#f87171', marginTop:'.5rem', textAlign:'center', lineHeight:1.4 }}>Camera denied.<br/>Check browser permissions.</div>
        : <div style={{ fontSize:'.65rem', color:'#64748b', marginTop:'.5rem', textAlign:'center' }}>{status}</div>
      }
    </div>
  )
}
