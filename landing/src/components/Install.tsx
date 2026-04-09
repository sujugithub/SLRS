import { Check, Copy, Terminal } from 'lucide-react'
import { useState } from 'react'

const cmd = `git clone https://github.com/sujugithub/SLRS.git
cd SLRS
pip install -r requirements.txt
python main.py`

export default function Install() {
  const [copied, setCopied] = useState(false)

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(cmd)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      /* noop */
    }
  }

  return (
    <section id="install" className="relative py-24">
      <div className="container-xl">
        <div className="mx-auto max-w-3xl">
          <div className="text-center">
            <p className="text-sm font-semibold uppercase tracking-wider text-cyan-300">
              Get started
            </p>
            <h2 className="mt-3 text-3xl font-extrabold tracking-tight md:text-4xl">
              Up and running in four commands.
            </h2>
            <p className="mt-4 text-slate-400">
              Requires Python 3.9+ and a working webcam. The app launches
              directly into the live prediction screen.
            </p>
          </div>

          <div className="mt-10 overflow-hidden rounded-2xl border border-white/10 bg-ink-900/80 shadow-2xl shadow-indigo-900/10 backdrop-blur">
            <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <Terminal className="h-4 w-4 text-cyan-300" />
                <span>bash</span>
              </div>
              <button
                onClick={onCopy}
                className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-200 transition hover:bg-white/10"
              >
                {copied ? (
                  <>
                    <Check className="h-3.5 w-3.5 text-emerald-300" /> Copied
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5" /> Copy
                  </>
                )}
              </button>
            </div>
            <pre className="overflow-x-auto p-5 text-sm leading-relaxed text-slate-200">
              <code>
                {cmd.split('\n').map((line, i) => (
                  <div key={i}>
                    <span className="select-none text-slate-600">$ </span>
                    {line}
                  </div>
                ))}
              </code>
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
