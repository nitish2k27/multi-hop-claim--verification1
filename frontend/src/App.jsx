import { useEffect, useState } from 'react'
import ClaimInput from './components/ClaimInput'
import History from './components/History'
import Progress from './components/Progress'
import Report from './components/Report'
import { getHealth, verifyStream } from './api'
import { clearHistory, loadHistory, removeEntry, saveEntry } from './history'

export default function App() {
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(null)
  const [entries, setEntries] = useState(() => loadHistory())
  const [activeId, setActiveId] = useState(null)
  const [steps, setSteps] = useState([])
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setHealthError(e.message))
  }, [])

  async function run({ claim, file }) {
    setBusy(true)
    setError(null)
    setResult(null)
    setSteps([])
    setActiveId(null)

    try {
      const final = await verifyStream({
        claim,
        file,
        onStep: (node, detail) => setSteps((prev) => [...prev, { node, detail }]),
      })
      setResult(final)
      const { entry, history } = saveEntry(final, { fileName: file?.name })
      setEntries(history)
      setActiveId(entry.id)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  function openEntry(entry) {
    // Replays a stored run. The steps panel is cleared rather than faked —
    // showing a fabricated progress trace for a result loaded from disk would
    // be a lie about what just happened.
    setResult(entry.result)
    setActiveId(entry.id)
    setSteps([])
    setError(null)
  }

  const caps = health?.capabilities || {}

  return (
    <div className="layout">
      <main className="main">
        <header>
          <h1>VerifAI</h1>
          <span className="dim small">
            {healthError
              ? `backend unreachable — ${healthError}`
              : health
                ? `index ${health.index} · ${(health.documents || 0).toLocaleString()} documents · floor ${health.relevance_floor}`
                : 'connecting…'}
          </span>
        </header>

        <p className="lede">
          Fact verification that knows when to say nothing. Paste a claim, a
          screenshot, a voice note or a document.
        </p>

        {health?.status === 'error' && <div className="error">{health.detail}</div>}

        <ClaimInput caps={caps} busy={busy} onSubmit={run} />

        <Progress steps={steps} busy={busy} />

        {error && <div className="error">{error}</div>}

        <Report result={result} />
      </main>

      <History
        entries={entries}
        activeId={activeId}
        onOpen={openEntry}
        onRemove={(id) => setEntries(removeEntry(id))}
        onClear={() => {
          setEntries(clearHistory())
          setActiveId(null)
        }}
      />
    </div>
  )
}
