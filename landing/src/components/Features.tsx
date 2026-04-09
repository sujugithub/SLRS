import {
  Camera,
  Trees,
  Waves,
  MessageSquareText,
  Volume2,
  HardDrive,
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
    body: 'Real-time hand, pose, and face landmark detection powered by MediaPipe Tasks — straight from your camera feed.',
  },
  {
    icon: Trees,
    title: 'Static Random Forest',
    body: 'A lightweight sklearn RandomForest classifier turns 126 flattened landmark features into an instant sign prediction.',
  },
  {
    icon: Waves,
    title: 'Dynamic MLP',
    body: 'A rolling 30-frame buffer feeds engineered temporal descriptors (mean, std, displacement, velocity) into an MLPClassifier.',
  },
  {
    icon: MessageSquareText,
    title: 'Sentence assembly',
    body: 'Phrase matching and lightweight NLP cleanup stitch individual signs into natural sentences as you sign.',
  },
  {
    icon: Volume2,
    title: 'Text-to-speech',
    body: 'Cross-platform TTS output speaks your sentences aloud with automatic fallback between native engines.',
  },
  {
    icon: HardDrive,
    title: 'Local & persistent',
    body: 'All training data, settings, phrase shortcuts, and model metadata are stored on disk — no accounts, no cloud.',
  },
]

export default function Features() {
  return (
    <section id="features" className="relative py-24">
      <div className="container-xl">
        <div className="mx-auto max-w-2xl text-center">
          <p className="text-sm font-semibold uppercase tracking-wider text-cyan-300">
            What's inside
          </p>
          <h2 className="mt-3 text-3xl font-extrabold tracking-tight md:text-4xl">
            Everything you need for on-device sign recognition.
          </h2>
          <p className="mt-4 text-slate-400">
            A focused desktop stack built for speed, privacy, and extensibility.
          </p>
        </div>

        <div className="mt-14 grid gap-5 md:grid-cols-2 lg:grid-cols-3">
          {features.map((f) => (
            <div key={f.title} className="card">
              <span className="inline-flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500/20 to-cyan-400/20 text-cyan-300 ring-1 ring-inset ring-white/10">
                <f.icon className="h-5 w-5" />
              </span>
              <h3 className="mt-5 text-lg font-semibold text-white">
                {f.title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">
                {f.body}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
