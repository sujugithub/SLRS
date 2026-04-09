const techs = [
  'Python 3.9+',
  'PyQt6',
  'MediaPipe Tasks',
  'scikit-learn',
  'OpenCV',
  'NumPy',
]

export default function TechStack() {
  return (
    <section className="py-16">
      <div className="container-xl">
        <p className="text-center text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
          Built with
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          {techs.map((t) => (
            <span
              key={t}
              className="rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-sm text-slate-300 backdrop-blur"
            >
              {t}
            </span>
          ))}
        </div>
      </div>
    </section>
  )
}
