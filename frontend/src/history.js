/**
 * Session history — every verification you have run, kept in the browser.
 *
 * localStorage rather than a backend table, deliberately: this is a
 * single-user local tool with no accounts, so there is nobody to scope a
 * server-side history to. It also means history survives a backend restart,
 * which matters because the backend restarts every time the corpus is rebuilt.
 *
 * Entries are capped and trimmed. A full result carries the rendered markdown
 * plus every evidence item, which is tens of kilobytes; localStorage gives you
 * about 5 MB total, so keeping 200 unbounded runs would silently start throwing
 * QuotaExceededError mid-save.
 */

const KEY = 'verifai.history.v1'
const MAX_ENTRIES = 40

export function loadHistory() {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    // Corrupt or unreadable: start clean rather than breaking the app.
    return []
  }
}

export function saveEntry(result, { fileName } = {}) {
  const entry = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    at: new Date().toISOString(),
    claim: result.claim || '',
    verdict: result.report?.verdict || null,
    confidence: result.report?.confidence ?? null,
    outcome: result.outcome,
    language: result.language || 'en',
    inputKind: result.input_kind || 'text',
    fileName: fileName || null,
    evidenceCount: result.evidence?.length || 0,
    llmCalls: result.diagnostics?.llm_calls ?? 0,
    elapsed: result.elapsed_seconds ?? null,
    result,
  }

  const history = [entry, ...loadHistory()].slice(0, MAX_ENTRIES)
  persist(history)
  return { entry, history }
}

export function removeEntry(id) {
  const history = loadHistory().filter((e) => e.id !== id)
  persist(history)
  return history
}

export function clearHistory() {
  localStorage.removeItem(KEY)
  return []
}

function persist(history) {
  try {
    localStorage.setItem(KEY, JSON.stringify(history))
  } catch {
    // Over quota. Drop the oldest half and try once more rather than losing
    // the run the user just did.
    try {
      localStorage.setItem(
        KEY,
        JSON.stringify(history.slice(0, Math.floor(history.length / 2)))
      )
    } catch {
      /* give up silently — history is a convenience, not the product */
    }
  }
}

export function relativeTime(iso) {
  const seconds = Math.round((Date.now() - new Date(iso)) / 1000)
  if (seconds < 60) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.round(hours / 24)}d ago`
}
