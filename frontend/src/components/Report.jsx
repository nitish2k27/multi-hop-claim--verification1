import { useState } from 'react'

const VERDICTS = {
  TRUE: ['TRUE', 'var(--true)'],
  MOSTLY_TRUE: ['MOSTLY TRUE', 'var(--mostly-true)'],
  UNVERIFIABLE: ['UNVERIFIABLE', 'var(--unverifiable)'],
  MOSTLY_FALSE: ['MOSTLY FALSE', 'var(--mostly-false)'],
  FALSE: ['FALSE', 'var(--false)'],
}

export default function Report({ result }) {
  const [showRaw, setShowRaw] = useState(false)
  if (!result) return null

  const report = result.report

  // Rejected input: there is no claim, so a verification of it would be a
  // category error. Show why instead.
  if (!report) {
    return (
      <section className="report">
        <p className="claim-line">{result.claim}</p>
        <div className="verdict" style={{ borderColor: 'var(--unverifiable)' }}>
          <b style={{ color: 'var(--unverifiable)' }}>NOT A CLAIM</b>
        </div>
        <pre className="raw">{result.rendered}</pre>
      </section>
    )
  }

  const [label, colour] = VERDICTS[report.verdict] || [report.verdict, 'var(--muted)']
  const abstained = result.outcome === 'abstained'

  return (
    <section className="report">
      <p className="claim-line">{report.claim}</p>

      <div className="verdict" style={{ borderColor: colour }}>
        <b style={{ color: colour }}>{label}</b>
        <span className="dim">{report.confidence}% confidence</span>
        <span className="dim">
          · {result.outcome} · {result.diagnostics?.llm_calls ?? 0} LLM call
          {(result.diagnostics?.llm_calls ?? 0) === 1 ? '' : 's'}
          {result.elapsed_seconds ? ` · ${result.elapsed_seconds}s` : ''}
        </span>
      </div>

      {abstained && (
        <p className="note">
          No evidence was relevant enough to judge this claim, so no verdict is
          offered. This is a statement about the available evidence, not about
          the claim.
        </p>
      )}

      {result.input_kind !== 'text' && (result.transcript || result.extracted_text) && (
        <>
          <h3>{result.input_kind === 'voice' ? 'Transcript' : 'Extracted text'}</h3>
          <div className="box">{result.transcript || result.extracted_text}</div>
        </>
      )}

      {report.key_findings?.length > 0 && (
        <>
          <h3>Key findings</h3>
          <ul>
            {report.key_findings.map((f, i) => (
              <li key={i}>
                {f.finding} <span className="dim">[{f.evidence_index}]</span>
                {f.quote && <blockquote>{f.quote}</blockquote>}
              </li>
            ))}
          </ul>
        </>
      )}

      {result.evidence?.length > 0 && (
        <>
          <h3>Evidence</h3>
          {result.evidence.map((e) => {
            const a = report.evidence_analysis?.find((x) => x.evidence_index === e.index)
            return <Evidence key={e.index} e={e} a={a} />
          })}
        </>
      )}

      {report.contradictions?.length > 0 && (
        <>
          <h3>Contradictions</h3>
          <ul>
            {report.contradictions.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </>
      )}

      {report.limitations?.length > 0 && (
        <>
          <h3>Limitations</h3>
          <ul>
            {report.limitations.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </>
      )}

      <h3>Conclusion</h3>
      <p>{report.conclusion}</p>

      {result.artifacts?.mp3 && (
        <>
          <h3>Listen</h3>
          <audio controls src={result.artifacts.mp3} />
        </>
      )}

      {Object.entries(result.artifacts || {}).filter(([k]) => k !== 'mp3').length > 0 && (
        <div className="downloads">
          {Object.entries(result.artifacts)
            .filter(([k]) => k !== 'mp3')
            .map(([k, v]) => (
              <a key={k} href={v} download>
                ⬇ {k}
              </a>
            ))}
        </div>
      )}

      <button className="link" onClick={() => setShowRaw((v) => !v)}>
        {showRaw ? 'Hide' : 'Read'} the full report
      </button>
      {showRaw && <pre className="raw">{result.rendered}</pre>}
    </section>
  )
}

function Evidence({ e, a }) {
  const pct = Math.round((e.credibility || 0) * 100)
  const isUpload = e.origin === 'user_upload'
  const isWeb = e.origin === 'web'

  return (
    <div className="ev">
      <div className="ev-top">
        <span className="dim">[{e.index}]</span>
        {e.url && !e.url.startsWith('upload://') ? (
          <a href={e.url} target="_blank" rel="noopener noreferrer">
            {e.domain || 'source'}
          </a>
        ) : (
          <strong>{e.domain || 'source'}</strong>
        )}
        {isUpload && <span className="badge warn">your upload · unverified</span>}
        {isWeb && <span className="badge web">web search</span>}
        {a && <span className="badge">{a.stance}</span>}
        <span className="dim meter-wrap">
          credibility
          <span className="meter">
            <i style={{ width: `${pct}%` }} />
          </span>
          {(e.credibility || 0).toFixed(2)}
        </span>
      </div>
      {a?.reasoning && <div>{a.reasoning}</div>}
      {e.title && <div className="dim small">{e.title}</div>}
    </div>
  )
}
