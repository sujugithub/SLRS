import { Github, Hand } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="border-t border-white/5 py-10">
      <div className="container-xl flex flex-col items-center justify-between gap-4 md:flex-row">
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <span className="grid h-7 w-7 place-items-center rounded-lg bg-gradient-to-br from-indigo-500 to-cyan-400 text-ink-950">
            <Hand className="h-4 w-4" strokeWidth={2.5} />
          </span>
          <span>
            SignBridge · Provided for educational purposes.
          </span>
        </div>
        <a
          href="https://github.com/sujugithub/SLRS"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-white"
        >
          <Github className="h-4 w-4" />
          sujugithub/SLRS
        </a>
      </div>
    </footer>
  )
}
