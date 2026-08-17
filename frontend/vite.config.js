import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// HOW THE FRONTEND TALKS TO FASTAPI
// ---------------------------------
// In development the React dev server runs on :5173 and FastAPI on :8000.
// Rather than enabling CORS and dealing with preflight requests on a
// multipart upload, Vite proxies the API paths straight through — so from the
// browser's point of view everything is same-origin and `fetch('/verify')`
// just works.
//
// In production `npm run build` writes ../ui/dist, which FastAPI serves itself.
// Same relative URLs, no proxy, no CORS, one process.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      ['/verify', '/health', '/download'].map((path) => [
        path,
        {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
          // SSE must not be buffered by the proxy or the progress panel
          // arrives all at once when the run finishes.
          configure: (proxy) => {
            proxy.on('proxyRes', (proxyRes) => {
              proxyRes.headers['cache-control'] = 'no-cache'
            })
          },
        },
      ])
    ),
  },
  build: {
    outDir: '../ui/dist',
    emptyOutDir: true,
  },
})
