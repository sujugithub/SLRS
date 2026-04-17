import { ArrowRight, Github } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

export default function Hero() {
  return (
    <section id="top" className="relative overflow-hidden border-b border-white/[0.06]">

      {/* Dot grid background */}
      <div className="pointer-events-none absolute inset-0 dot-grid opacity-60" />

      {/* Amber radial glow top */}
      <div className="pointer-events-none absolute inset-0 amber-glow-bg" />

      <div className="container-xl relative grid items-center gap-16 py-24 md:grid-cols-2 md:py-32">

        {/* ── Left column ── */}
        <div className="flex flex-col">
          <span className="label-overline mb-6">
            Local-first · No cloud · Open source
          </span>

          <h1 className="font-display text-5xl font-extrabold leading-[1.04] tracking-tight text-white md:text-[4rem]">
            Real-time<br />
            <span className="text-amber-400">sign language</span><br />
            recognition.
          </h1>

          <p className="mt-6 max-w-lg text-[15px] leading-relaxed text-zinc-400">
            SignBridge reads hand, pose, and face landmarks from your webcam,
            classifies static and dynamic signs using ML, and speaks them aloud —
            entirely on your machine.
          </p>

          <div className="mt-8 flex flex-wrap gap-3">
            <a href="#install" className="btn-primary">
              Get started <ArrowRight className="h-4 w-4" />
            </a>
            <a
              href="https://github.com/sujugithub/SLRS"
              target="_blank"
              rel="noreferrer"
              className="btn-ghost"
            >
              <Github className="h-4 w-4" />
              View on GitHub
            </a>
          </div>

          {/* Stats strip */}
          <div className="mt-12 grid grid-cols-3 divide-x divide-white/[0.06] overflow-hidden rounded-2xl border border-white/[0.06]">
            {[
              { n: '21',  label: 'Landmarks / hand' },
              { n: '30',  label: 'Frames / sequence' },
              { n: '2',   label: 'Hands tracked' },
            ].map(({ n, label }) => (
              <div key={label} className="bg-ink-900 px-5 py-4">
                <div className="font-display text-2xl font-bold text-amber-400">{n}</div>
                <div className="mt-0.5 text-[11px] text-zinc-600">{label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Right column — Photo Animation ── */}
        <PhotoFlipbook />
      </div>
    </section>
  )
}

/* ──────────────────────────────────────────────── */
/*  Animated photo flipbook                        */
/* ──────────────────────────────────────────────── */

const FRAMES = [
  { src: '/screenshots/drink.png',      label: 'DRINK',      conf: '94%', color: '#f59e0b', tilt: '-2deg'  },
  { src: '/screenshots/hello.png',      label: 'HELLO',      conf: '99%', color: '#10b981', tilt:  '1.5deg' },
  { src: '/screenshots/i-love-you.png', label: 'I LOVE YOU', conf: '97%', color: '#ec4899', tilt: '-1deg'  },
  { src: '/screenshots/sorry.png',      label: 'SORRY',      conf: '92%', color: '#a78bfa', tilt:  '2deg'  },
  { src: '/screenshots/thank-you.png',  label: 'THANK YOU',  conf: '98%', color: '#f59e0b', tilt: '-1.5deg' },
  { src: '/screenshots/you.png',        label: 'YOU',        conf: '96%', color: '#38bdf8', tilt:  '1deg'  },
]

// tiny decorative particles shown around the card on each frame
const SPARKS = [
  { top: '8%',  left: '92%',  size: 10, delay: 0    },
  { top: '20%', left: '-6%',  size: 7,  delay: 0.15 },
  { top: '75%', left: '95%',  size: 8,  delay: 0.08 },
  { top: '88%', left: '-4%',  size: 6,  delay: 0.22 },
  { top: '50%', left: '97%',  size: 5,  delay: 0.3  },
  { top: '35%', left: '-8%',  size: 9,  delay: 0.05 },
]

function PhotoFlipbook() {
  const [idx,    setIdx]    = useState(0)
  const [burst,  setBurst]  = useState(false)
  const prevIdx = useRef(0)

  // Cycle frames at ~550 ms each
  useEffect(() => {
    const id = setInterval(() => {
      setIdx(i => (i + 1) % FRAMES.length)
      setBurst(true)
      setTimeout(() => setBurst(false), 160)
    }, 550)
    return () => clearInterval(id)
  }, [])

  useEffect(() => { prevIdx.current = idx }, [idx])

  const frame = FRAMES[idx]

  return (
    <div className="relative flex items-center justify-center">

      {/* Ambient color glow that shifts with the sign */}
      <div
        className="pointer-events-none absolute inset-0 -z-10 flex items-center justify-center transition-all duration-500"
      >
        <div
          className="h-80 w-80 rounded-full blur-3xl transition-all duration-500"
          style={{ background: frame.color + '22' }}
        />
      </div>

      {/* Floating sparkle dots */}
      {SPARKS.map((s, i) => (
        <span
          key={i}
          className="pointer-events-none absolute select-none"
          style={{
            top: s.top,
            left: s.left,
            fontSize: s.size,
            opacity: burst ? 1 : 0.35,
            transform: burst ? 'scale(1.6)' : 'scale(1)',
            transition: `opacity 0.15s ${s.delay}s, transform 0.15s ${s.delay}s`,
            color: frame.color,
            fontWeight: 900,
            lineHeight: 1,
          }}
        >
          ✦
        </span>
      ))}

      {/* Card */}
      <div
        className="relative w-full max-w-sm overflow-hidden rounded-3xl shadow-2xl shadow-black/70"
        style={{
          border: `2.5px solid ${frame.color}55`,
          transform: `rotate(${frame.tilt})`,
          transition: 'transform 0.2s cubic-bezier(.34,1.56,.64,1), border-color 0.3s',
          background: '#0a0a0c',
        }}
      >
        {/* Window chrome */}
        <div className="flex items-center gap-2 px-4 pt-3 pb-2">
          <span className="h-2.5 w-2.5 rounded-full bg-red-500/70" />
          <span className="h-2.5 w-2.5 rounded-full bg-yellow-400/70" />
          <span className="h-2.5 w-2.5 rounded-full bg-green-500/70" />
          <span className="ml-3 font-mono text-[11px] text-zinc-600">signbridge · live</span>
          {/* frame counter */}
          <span className="ml-auto font-mono text-[10px] text-zinc-700">#{(idx + 1).toString().padStart(4, '0')}</span>
        </div>

        {/* Photo */}
        <div className="relative aspect-[4/3] overflow-hidden bg-zinc-900">
          {FRAMES.map((f, i) => (
            <img
              key={f.src}
              src={f.src}
              alt={f.label}
              className="absolute inset-0 h-full w-full object-cover object-[50%_35%]"
              style={{
                opacity: i === idx ? 1 : 0,
                transform: i === idx
                  ? 'scale(1.02)'
                  : (i === prevIdx.current ? 'scale(0.96)' : 'scale(1)'),
                transition: 'opacity 0.12s ease, transform 0.2s ease',
                zIndex: i === idx ? 1 : 0,
              }}
            />
          ))}

          {/* Scanline overlay */}
          <div className="pointer-events-none absolute inset-0 z-10"
               style={{ background: 'repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.08) 2px,rgba(0,0,0,0.08) 4px)' }} />

          {/* Live badge */}
          <div className="absolute left-3 top-3 z-20 flex items-center gap-1.5 rounded-full bg-black/60 px-3 py-1 font-mono text-[11px] font-semibold text-white backdrop-blur">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-red-400" />
            LIVE FEED
          </div>

          {/* Frame number badge */}
          <div className="absolute right-3 top-3 z-20 rounded-full bg-black/60 px-3 py-1 font-mono text-[11px] text-zinc-400 backdrop-blur">
            #{(Math.floor(Math.random() * 999) + 1000)}
          </div>

          {/* Sign label pop — big & fun */}
          <div
            className="absolute bottom-0 inset-x-0 z-20 flex flex-col items-start px-4 pb-3 pt-6"
            style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.82) 0%, transparent 100%)' }}
          >
            <span
              className="font-display text-4xl font-extrabold tracking-tight leading-none"
              style={{
                color: frame.color,
                textShadow: `0 0 18px ${frame.color}88`,
                transform: burst ? 'scale(1.08)' : 'scale(1)',
                display: 'inline-block',
                transition: 'transform 0.12s cubic-bezier(.34,1.56,.64,1), color 0.2s',
              }}
            >
              {frame.label}
            </span>
            <span className="mt-0.5 font-mono text-[11px] text-zinc-500">{frame.conf}</span>
          </div>
        </div>

        {/* Bottom strip — cycling dots */}
        <div className="flex items-center justify-center gap-2 py-3">
          {FRAMES.map((_f, i) => (
            <span
              key={i}
              className="rounded-full transition-all duration-200"
              style={{
                width:  i === idx ? 18 : 6,
                height: 6,
                background: i === idx ? frame.color : 'rgba(255,255,255,0.1)',
              }}
            />
          ))}
        </div>
      </div>

      {/* Comic-style "speech" burst that pops on frame change */}
      <div
        className="pointer-events-none absolute -top-4 -right-2 z-30 select-none font-display text-[11px] font-extrabold tracking-widest uppercase"
        style={{
          color: frame.color,
          opacity: burst ? 1 : 0,
          transform: burst ? 'scale(1) rotate(-8deg)' : 'scale(0.5) rotate(-8deg)',
          transition: 'opacity 0.1s, transform 0.12s cubic-bezier(.34,1.56,.64,1)',
          textShadow: `0 0 12px ${frame.color}`,
        }}
      >
        ✦ {frame.label}!
      </div>

    </div>
  )
}
