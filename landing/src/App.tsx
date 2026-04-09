import Nav from './components/Nav'
import Hero from './components/Hero'
import Features from './components/Features'
import Pipeline from './components/Pipeline'
import TechStack from './components/TechStack'
import Install from './components/Install'
import Footer from './components/Footer'
import GestureController from './components/GestureController'

export default function App() {
  return (
    <div className="min-h-screen bg-ink-950 text-slate-100">
      <Nav />
      <main>
        <Hero />
        <Features />
        <Pipeline />
        <TechStack />
        <Install />
      </main>
      <Footer />
      <GestureController />
    </div>
  )
}
