import { relativeTime } from '../history'

const DOT = {
  TRUE: 'var(--true)',
  MOSTLY_TRUE: 'var(--mostly-true)',
  UNVERIFIABLE: 'var(--unverifiable)',
  MOSTLY_FALSE: 'var(--mostly-false)',
  FALSE: 'var(--false)',
}

const KIND_ICON = { text: '', voice: '🎙 ', image: '🖼 ', document: '📄 ' }

export default function History({ entries, activeId, onOpen, onRemove, onClear }) {
  return (
    <aside className="history">
      <div className="history-head">
        <h2>History</h2>
        {entries.length > 0 && (
          <button className="link" onClick={onClear}>
            clear
          </button>
        )}
      </div>

      {entries.length === 0 ? (
        <p className="dim small">
          Verifications you run are kept here, in this browser only. Nothing is
          sent anywhere or stored on a server.
        </p>
      ) : (
        <ul className="history-list">
          {entries.map((e) => (
            <li key={e.id} className={e.id === activeId ? 'active' : undefined}>
              <button className="history-item" onClick={() => onOpen(e)}>
                <span
                  className="pip"
                  style={{ background: DOT[e.verdict] || 'var(--line)' }}
                  title={e.verdict || e.outcome}
                />
                <span className="history-claim">
                  {KIND_ICON[e.inputKind] || ''}
                  {e.claim || e.fileName || '(no claim)'}
                </span>
                <span className="dim small">
                  {e.outcome === 'rejected'
                    ? 'not a claim'
                    : e.outcome === 'abstained'
                      ? 'abstained'
                      : `${(e.verdict || '').replace('_', ' ')} ${e.confidence ?? ''}%`}
                  {' · '}
                  {relativeTime(e.at)}
                </span>
              </button>
              <button className="x" title="remove" onClick={() => onRemove(e.id)}>
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
    </aside>
  )
}
