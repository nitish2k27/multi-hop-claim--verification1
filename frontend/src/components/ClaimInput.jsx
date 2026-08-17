import { useEffect, useRef, useState } from 'react'

const EXAMPLES = [
  { label: 'verifiable', text: "India's software exports reached 222 billion dollars in 2024-25" },
  { label: 'on-topic, absent', text: "India's GDP grew 8% in 2024" },
  { label: 'should abstain', text: 'the moon is square in shape' },
  { label: 'not a claim', text: 'what time is it' },
  { label: 'Hindi', text: 'भारत का सॉफ्टवेयर निर्यात 2024-25 में 222 अरब डॉलर तक पहुंच गया' },
]

export default function ClaimInput({ caps, busy, onSubmit }) {
  const [claim, setClaim] = useState('')
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef(null)

  // Paste a screenshot straight from the clipboard. This is the whole point of
  // image input: see a claim in a post, screenshot it, Ctrl+V, done.
  useEffect(() => {
    function onPaste(event) {
      const item = [...(event.clipboardData?.items || [])].find((i) =>
        i.type.startsWith('image/')
      )
      if (!item) return
      const blob = item.getAsFile()
      if (!blob) return
      event.preventDefault()
      const ext = (blob.type.split('/')[1] || 'png').replace('jpeg', 'jpg')
      setFile(new File([blob], `pasted-screenshot.${ext}`, { type: blob.type }))
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [])

  function submit() {
    if (busy || (!claim.trim() && !file)) return
    onSubmit({ claim: claim.trim(), file })
  }

  // Three distinct behaviours, and it is worth telling the user which one they
  // are about to get — attaching a file to a typed claim means something
  // different from attaching one alone.
  const mode = !file
    ? 'Verifies the text you typed.'
    : claim.trim()
      ? 'Verifies your claim against the attached file. The file is capped at 0.6 credibility — an upload cannot make its own claim true.'
      : 'Extracts a claim from the file, then verifies it.'

  return (
    <section className="panel">
      <textarea
        className="claim-box"
        value={claim}
        placeholder={'Type or paste a claim…\ne.g. India\'s software exports reached 222 billion dollars in 2024-25'}
        onChange={(e) => setClaim(e.target.value)}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') submit()
        }}
        disabled={busy}
      />

      <div
        className={`drop${dragging ? ' over' : ''}`}
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0])
        }}
      >
        <strong>Drop a file</strong>, click to browse, or <strong>paste a screenshot</strong> (Ctrl+V)
        <div className="caps">
          <Cap on={caps.voice} label="voice" />
          <Cap on={caps.image} label="image" hint={caps.image_reason} />
          <Cap on label="pdf / docx" />
          <Cap on={caps.web_search} label="web fallback" />
        </div>
      </div>

      <input
        ref={fileRef}
        type="file"
        hidden
        accept=".mp3,.wav,.m4a,.ogg,.webm,.flac,.png,.jpg,.jpeg,.webp,.pdf,.docx,.txt,.md"
        onChange={(e) => e.target.files[0] && setFile(e.target.files[0])}
      />

      {file && (
        <div className="chip-file">
          <span>📎 {file.name}</span>
          <span className="dim">{(file.size / 1024).toFixed(0)} KB</span>
          <button
            className="x"
            onClick={() => {
              setFile(null)
              if (fileRef.current) fileRef.current.value = ''
            }}
            title="remove"
          >
            ✕
          </button>
        </div>
      )}

      <div className="actions">
        <button className="primary" onClick={submit} disabled={busy || (!claim.trim() && !file)}>
          {busy ? 'Verifying…' : 'Verify'}
        </button>
        <span className="dim mode">{mode}</span>
      </div>

      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            className="chip"
            disabled={busy}
            onClick={() => {
              setClaim(ex.text)
              setFile(null)
            }}
          >
            {ex.label}
          </button>
        ))}
      </div>
    </section>
  )
}

function Cap({ on, label, hint }) {
  return (
    <span className={`cap${on ? ' on' : ''}`} title={!on && hint ? hint : undefined}>
      {on ? '✓' : '✗'} {label}
    </span>
  )
}
