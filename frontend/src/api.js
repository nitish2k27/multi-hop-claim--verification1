/**
 * Talking to the FastAPI backend.
 *
 * Relative URLs throughout. In dev, Vite proxies them to :8000 (see
 * vite.config.js); in a build, FastAPI serves this bundle itself, so the same
 * URLs are already same-origin. Nothing here needs to know which it is.
 */

export async function getHealth() {
  const res = await fetch('/health')
  if (!res.ok) throw new Error(`Backend returned ${res.status}`)
  return res.json()
}

/**
 * Run a verification, streaming progress as it happens.
 *
 * The backend emits one server-sent event per completed graph node, so
 * `onStep` fires roughly seven times for a full run. That is the whole reason
 * the pipeline is a graph: node names arrive as step IDs for free.
 *
 * `fetch` is used rather than `EventSource` because EventSource cannot POST,
 * and this endpoint takes a multipart body with an optional file upload.
 */
export async function verifyStream({ claim, file, onStart, onStep, signal }) {
  const body = new FormData()
  body.append('claim', claim || '')
  if (file) body.append('file', file)

  const res = await fetch('/verify/stream', { method: 'POST', body, signal })
  if (!res.ok) throw new Error(`Backend returned ${res.status}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let result = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // SSE frames are separated by a blank line. Keep the trailing partial
    // frame in the buffer — a chunk boundary can land mid-event.
    const frames = buffer.split('\n\n')
    buffer = frames.pop()

    for (const frame of frames) {
      const line = frame.split('\n').find((l) => l.startsWith('data: '))
      if (!line) continue

      let event
      try {
        event = JSON.parse(line.slice(6))
      } catch {
        continue
      }

      if (event.type === 'start') onStart?.()
      else if (event.type === 'step') onStep?.(event.step, event.detail)
      else if (event.type === 'done') result = event.result
      else if (event.type === 'error') throw new Error(event.message)
    }
  }

  if (!result) throw new Error('Stream ended before a result arrived')
  return result
}
