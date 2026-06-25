// THREADS/BROWSER step — a tiny static server that sets the **cross-origin isolation** headers
// (`COOP: same-origin` + `COEP: require-corp`) a real browser requires before it will expose
// `SharedArrayBuffer` / shared `WebAssembly.Memory`. (Node's worker_threads need none of this; it is
// exactly the browser-only constraint this slice exists to exercise.) Serves the `browser/` tree so the
// page can fetch the wasm, the corpus, and the worker module. Exported for the Playwright harness;
// also runnable standalone: `node serve.mjs [root] [port]`.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const TYPES = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
  '.svmbc': 'application/octet-stream',
  '.json': 'application/json',
};

export function startServer(root, port = 0) {
  const server = createServer(async (req, res) => {
    const headers = {
      // The trio that makes `crossOriginIsolated === true`, unlocking SharedArrayBuffer.
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Resource-Policy': 'same-origin',
    };
    try {
      const url = new URL(req.url, 'http://localhost');
      let path = decodeURIComponent(url.pathname);
      if (path === '/') path = '/web/index.html';
      const file = normalize(join(root, path));
      if (!file.startsWith(normalize(root))) {
        res.writeHead(403, headers);
        return void res.end('forbidden');
      }
      const body = await readFile(file);
      headers['Content-Type'] = TYPES[extname(file)] ?? 'application/octet-stream';
      res.writeHead(200, headers);
      res.end(body);
    } catch {
      res.writeHead(404, headers);
      res.end('not found');
    }
  });
  return new Promise((resolve) => {
    server.listen(port, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const root = process.argv[2] ?? '.';
  const port = Number(process.argv[3] ?? 8088);
  const { port: p } = await startServer(root, port);
  console.log(`serving ${root} on http://127.0.0.1:${p} (COOP/COEP cross-origin isolated)`);
}
