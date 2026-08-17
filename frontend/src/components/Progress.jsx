/**
 * Live progress — one row per completed graph node.
 *
 * The labels are friendlier than the node names, but the mapping is 1:1 with
 * the backend graph on purpose: if a node is added to the pipeline, a row
 * appears here with no other change.
 */

const LABELS = {
  adapt: 'read input',
  translate: 'language',
  gate: 'claim check',
  retrieve: 'search index',
  web: 'web search',
  stance: 'stance',
  generate: 'verdict',
  abstain: 'abstain',
  reject: 'reject',
  render: 'report',
}

export function describe(node, d = {}) {
  switch (node) {
    case 'adapt':
      if (d.error) return d.error.split('\n')[0]
      return d.kind === 'text'
        ? 'text'
        : `${d.kind} → ${(d.extracted || '').length} characters extracted`
    case 'translate':
      return d.language === 'en' ? 'English — no translation needed' : `${d.language} → English`
    case 'gate':
      return d.is_claim
        ? `is a verifiable claim (${((d.confidence ?? 0) * 100).toFixed(1)}%)`
        : 'not a verifiable claim'
    case 'retrieve':
    case 'web':
      return `${d.candidates ?? 0} candidates → ${d.evidence ?? 0} above floor ${d.floor ?? ''}`
    case 'stance':
      return Object.entries(d)
        .map(([k, v]) => `${k} ${v}`)
        .join(' · ')
    case 'generate':
      return `${d.verdict} at ${d.confidence}%`
    case 'abstain':
    case 'reject':
      return 'no language model called'
    case 'render':
      return `${d.chars} characters${d.artifacts?.length ? ` · ${d.artifacts.join(', ')}` : ''}`
    default:
      return ''
  }
}

export default function Progress({ steps, busy }) {
  if (!steps.length) return null

  return (
    <section className="panel steps">
      {steps.map((s, i) => (
        <div className="step" key={`${s.node}-${i}`}>
          <span className="dot" />
          <span className="step-name">{LABELS[s.node] || s.node}</span>
          <span className="dim">{describe(s.node, s.detail)}</span>
        </div>
      ))}
      {busy && (
        <div className="step pending">
          <span className="dot pulse" />
          <span className="dim">working…</span>
        </div>
      )}
    </section>
  )
}
