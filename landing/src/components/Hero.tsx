import { ArrowRight, Github, Sparkles } from 'lucide-react'

export default function Hero() {
  return (
    <section id="top" className="grid-bg relative overflow-hidden">
      <div className="container-xl relative grid gap-14 py-20 md:grid-cols-[1.1fr_0.9fr] md:py-28">
        <div className="flex flex-col justify-center">
          <span className="mb-6 inline-flex w-fit items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-slate-300 backdrop-blur">
            <Sparkles className="h-3.5 w-3.5 text-cyan-300" />
            Local-first · Runs entirely on your machine
          </span>

          <h1 className="text-4xl font-extrabold leading-[1.05] tracking-tight md:text-6xl">
            Real-time <span className="gradient-text">sign language</span>
            <br />
            recognition, on your desktop.
          </h1>

          <p className="mt-6 max-w-xl text-lg leading-relaxed text-slate-400">
            SignBridge reads hand, pose, and face landmarks from your webcam,
            classifies static and dynamic signs, and speaks them out loud —
            all with no cloud round-trip.
          </p>

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <a href="#install" className="btn-primary">
              Get started
              <ArrowRight className="h-4 w-4" />
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

          <dl className="mt-10 grid max-w-md grid-cols-3 gap-6">
            <Stat label="Frames / sequence" value="30" />
            <Stat label="Feature width" value="171" />
            <Stat label="Hands tracked" value="2" />
          </dl>
        </div>

        <HeroVisual />
      </div>
    </section>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wider text-slate-500">
        {label}
      </dt>
      <dd className="mt-1 text-2xl font-bold gradient-text">{value}</dd>
    </div>
  )
}

function HeroVisual() {
  return (
    <div className="relative flex items-center justify-center">
      <div className="absolute inset-0 -z-10 blur-3xl">
        <div className="mx-auto h-72 w-72 rounded-full bg-indigo-500/30" />
      </div>

      <div className="relative w-full max-w-md rounded-3xl border border-white/10 bg-ink-900/80 p-4 shadow-2xl shadow-indigo-900/30 backdrop-blur">
        <div className="mb-3 flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-red-400/80" />
          <span className="h-2.5 w-2.5 rounded-full bg-yellow-300/80" />
          <span className="h-2.5 w-2.5 rounded-full bg-green-400/80" />
          <span className="ml-3 text-xs text-slate-500">
            signbridge · live
          </span>
        </div>

        <div className="relative aspect-[4/3] overflow-hidden rounded-2xl bg-gradient-to-br from-ink-800 to-ink-950">
          <svg
            viewBox="0 0 400 300"
            className="absolute inset-0 h-full w-full"
          >
            <defs>
              <linearGradient id="strokeGrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stopColor="#818cf8" />
                <stop offset="1" stopColor="#22d3ee" />
              </linearGradient>
            </defs>

            <g stroke="url(#strokeGrad)" strokeWidth="2" fill="none">
              <path d="M200 230 L200 150" />
              <path d="M200 150 L160 90" />
              <path d="M200 150 L185 80" />
              <path d="M200 150 L210 75" />
              <path d="M200 150 L235 82" />
              <path d="M200 150 L255 100" />
            </g>

            {[
              [200, 230],
              [200, 150],
              [160, 90],
              [185, 80],
              [210, 75],
              [235, 82],
              [255, 100],
            ].map(([x, y], i) => (
              <circle
                key={i}
                cx={x}
                cy={y}
                r="4"
                fill="#fff"
                stroke="url(#strokeGrad)"
                strokeWidth="2"
              />
            ))}
          </svg>

          <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-black/40 px-3 py-1 text-xs text-slate-200 backdrop-blur">
            <span className="h-2 w-2 animate-pulse rounded-full bg-red-400" />
            REC
          </div>
          <div className="absolute right-3 top-3 rounded-full bg-black/40 px-3 py-1 text-xs text-slate-200 backdrop-blur">
            30 FPS
          </div>
        </div>

        <div className="mt-4 rounded-xl border border-white/10 bg-black/30 p-3">
          <p className="text-[11px] uppercase tracking-wider text-slate-500">
            Prediction
          </p>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold text-white">"Hello"</span>
            <span className="text-sm font-semibold text-cyan-300">
              98.2% conf
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
