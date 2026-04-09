import { Github, Hand } from 'lucide-react'

export default function Nav() {
  return (
    <header className="sticky top-0 z-50 border-b border-white/5 bg-ink-950/70 backdrop-blur-xl">
      <div className="container-xl flex h-16 items-center justify-between">
        <a href="#top" className="flex items-center gap-2">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-400 text-ink-950">
            <Hand className="h-5 w-5" strokeWidth={2.5} />
          </span>
          <span className="text-base font-bold tracking-tight">
            Sign<span className="gradient-text">Bridge</span>
          </span>
        </a>

        <nav className="hidden items-center gap-8 text-sm text-slate-300 md:flex">
          <a href="#features" className="hover:text-white">
            Features
          </a>
          <a href="#pipeline" className="hover:text-white">
            Pipeline
          </a>
          <a href="#install" className="hover:text-white">
            Install
          </a>
        </nav>

        <a
          href="https://github.com/sujugithub/SLRS"
          target="_blank"
          rel="noreferrer"
          className="btn-ghost !py-2 !px-4"
        >
          <Github className="h-4 w-4" />
          <span className="hidden sm:inline">GitHub</span>
        </a>
      </div>
    </header>
  )
}
