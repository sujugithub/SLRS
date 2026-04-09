const staticSteps = [
  'The hand detector extracts up to two hands of landmarks.',
  'Static features are flattened to 126 values.',
  'A RandomForestClassifier predicts the most likely sign.',
  'The result is filtered by the configured confidence threshold.',
]

const dynamicSteps = [
  'The holistic detector produces a per-frame feature vector from hand, pose, and face landmarks.',
  'A rolling sequence buffer keeps the last 30 frames.',
  'Temporal descriptors are computed per feature: mean, std, displacement, and velocity.',
  'Engineered features are fed into an MLPClassifier.',
  'The best prediction is emitted when it clears the confidence gate.',
]

export default function Pipeline() {
  return (
    <section
      id="pipeline"
      className="relative border-y border-white/5 bg-ink-900/40 py-24"
    >
      <div className="container-xl">
        <div className="mx-auto max-w-2xl text-center">
          <p className="text-sm font-semibold uppercase tracking-wider text-cyan-300">
            Recognition pipeline
          </p>
          <h2 className="mt-3 text-3xl font-extrabold tracking-tight md:text-4xl">
            Two classifiers. One smooth experience.
          </h2>
          <p className="mt-4 text-slate-400">
            Static signs go through a fast Random Forest. Motion-heavy gestures
            flow through a temporal MLP pipeline.
          </p>
        </div>

        <div className="mt-14 grid gap-6 md:grid-cols-2">
          <PipelineCard title="Static signs" accent="indigo" steps={staticSteps} />
          <PipelineCard title="Dynamic gestures" accent="cyan" steps={dynamicSteps} />
        </div>
      </div>
    </section>
  )
}

function PipelineCard({
  title,
  accent,
  steps,
}: {
  title: string
  accent: 'indigo' | 'cyan'
  steps: string[]
}) {
  const ring =
    accent === 'indigo'
      ? 'from-indigo-500/30 to-indigo-500/0'
      : 'from-cyan-400/30 to-cyan-400/0'

  return (
    <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/[0.02] p-8">
      <div
        className={`pointer-events-none absolute -right-10 -top-10 h-40 w-40 rounded-full bg-gradient-to-br ${ring} blur-2xl`}
      />
      <h3 className="text-xl font-bold text-white">{title}</h3>
      <ol className="mt-6 space-y-4">
        {steps.map((step, i) => (
          <li key={i} className="flex gap-4">
            <span className="mt-0.5 grid h-7 w-7 flex-none place-items-center rounded-full border border-white/10 bg-white/5 text-xs font-semibold text-cyan-300">
              {i + 1}
            </span>
            <span className="text-sm leading-relaxed text-slate-300">
              {step}
            </span>
          </li>
        ))}
      </ol>
    </div>
  )
}
