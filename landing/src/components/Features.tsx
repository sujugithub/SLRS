import {
  Camera, Trees, Waves, MessageSquareText, Volume2, HardDrive,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

type Feature = {
  icon: LucideIcon
  title: string
  body: string
}

const features: Feature[] = [
  {
    icon: Camera,
    title: 'Webcam inference',
    body: 'Real-time hand, pose, and face landmark detection powered by MediaPipe Tasks — straight from your camera feed, zero latency.',
  },
  {
    icon: Trees,
    title: 'Static Random Forest',
    body: 'A lightweight sklearn RandomForest turns 126 flattened landmark features into an instant sign prediction.',
  },
  {
    icon: Waves,
    title: 'Dynamic MLP',
    body: 'A rolling 30-frame buffer feeds engineered temporal descriptors — mean, std, displacement, velocity — into an MLPClassifier.',
  },
  {
    icon: MessageSquareText,
    title: 'Sentence assembly',
    body: 'Phrase matching and lightweight NLP cleanup stitch individual signs into natural sentences as you sign.',
  },
  {
    icon: Volume2,
    title: 'Text-to-speech',
    body: 'Cross-platform TTS speaks your sentences aloud with automatic fallback between native engines.',
  },
  {
    icon: HardDrive,
    title: 'Local & persistent',
    body: 'All training data, settings, phrase shortcuts, and model metadata stored on disk. No accounts, no cloud.',
  },
]

export default function Features() {
  const Feature0Icon = features[0].icon
  const Feature1Icon = features[1].icon
  const Feature5Icon = features[5].icon

  return (
    <section id="features" className="relative py-28">
      <div className="container-xl">

        {/* Heading */}
        <div className="flex items-end justify-between border-b border-white/[0.06] pb-8">
          <div>
            <span className="label-overline">What's inside</span>
            <h2 className="section-heading mt-3">
              Everything you need<br className="hidden md:block" /> for on-device recognition.
            </h2>
          </div>
          <span className="hidden font-display text-[6rem] font-extrabold leading-none text-white/[0.04] md:block">
            06
          </span>
        </div>

        {/* Bento grid */}
        <div className="mt-8 grid grid-cols-1 gap-4 md:grid-cols-6">

          {/* Feature 0 — main card, wider */}
          <div className="card relative overflow-hidden md:col-span-4 md:min-h-52">
            <span className="feat-num">01</span>
            <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-amber-500/20 bg-amber-500/10 text-amber-400">
              <Feature0Icon className="h-5 w-5" />
            </span>
            <h3 className="mt-4 font-display text-xl font-bold text-white">{features[0].title}</h3>
            <p className="mt-2 max-w-sm text-sm leading-relaxed text-zinc-400">{features[0].body}</p>
            {/* Decorative live dot */}
            <div className="mt-4 flex items-center gap-2">
              <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
              <span className="font-mono text-[11px] text-zinc-600">camera stream active</span>
            </div>
          </div>

          {/* Feature 1 */}
          <div className="card relative overflow-hidden md:col-span-2">
            <span className="feat-num">02</span>
            <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-amber-500/20 bg-amber-500/10 text-amber-400">
              <Feature1Icon className="h-5 w-5" />
            </span>
            <h3 className="mt-4 font-display text-lg font-bold text-white">{features[1].title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-zinc-400">{features[1].body}</p>
          </div>

          {/* Features 2, 3, 4 — equal thirds */}
          {features.slice(2, 5).map((f, i) => (
            <div key={f.title} className="card relative overflow-hidden md:col-span-2">
              <span className="feat-num">0{i + 3}</span>
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-amber-500/20 bg-amber-500/10 text-amber-400">
                <f.icon className="h-5 w-5" />
              </span>
              <h3 className="mt-4 font-display text-lg font-bold text-white">{f.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-zinc-400">{f.body}</p>
            </div>
          ))}

          {/* Feature 5 — full width footer card */}
          <div className="card relative flex flex-col gap-4 overflow-hidden border-amber-500/15 bg-gradient-to-r from-ink-900 to-ink-800 md:col-span-6 md:flex-row md:items-center md:gap-8">
            <span className="feat-num" style={{ fontSize: '4rem', top: '-0.4rem' }}>06</span>
            <span className="inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border border-amber-500/25 bg-amber-500/12 text-amber-400">
              <Feature5Icon className="h-6 w-6" />
            </span>
            <div>
              <h3 className="font-display text-xl font-bold text-white">{features[5].title}</h3>
              <p className="mt-1 text-sm leading-relaxed text-zinc-400">{features[5].body}</p>
            </div>
            <div className="md:ml-auto">
              <div className="flex items-center gap-2 rounded-xl border border-white/[0.06] bg-ink-950/60 px-4 py-2.5">
                <span className="h-2 w-2 rounded-full bg-amber-400" />
                <span className="font-mono text-[11px] text-zinc-500">data stored locally</span>
              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  )
}
