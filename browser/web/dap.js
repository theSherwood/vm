// Minimal **Debug Adapter Protocol client over the wasm FFI** — the browser half of DEBUGGING.md's
// DAP-over-bytecode path. The cdylib hosts the real `svm-dap` server (selecting the bytecode backend);
// this just marshals a request JSON in and reads the reply JSON (a `[response, event…]` array) back,
// exactly like a DAP editor's transport but with no wire framing. UI only — it never touches the
// sandbox or any authority.
//
// `createDapClient(ex, memory)` starts a fresh session (`svm_dap_reset`) and returns `{ send }`.
// `send(command, args)` returns `{ response, events, all }` for the one request.
export function createDapClient(ex, memory) {
  let seq = 0;
  const enc = new TextEncoder();
  const dec = new TextDecoder();
  ex.svm_dap_reset(); // begin a clean session

  const send = (command, args = {}) => {
    const req = { seq: ++seq, type: 'request', command, arguments: args };
    const bytes = enc.encode(JSON.stringify(req));
    // Write the request into wasm memory, hand it to the server, then read the stashed reply. Take a
    // *fresh* memory view after the call — servicing the request may grow (detach) the linear memory.
    const p = ex.svm_alloc(bytes.length);
    new Uint8Array(memory.buffer).set(bytes, p);
    const rc = ex.svm_dap_request(p, bytes.length);
    ex.svm_dealloc(p, bytes.length);
    if (rc !== 0) throw new Error(`DAP request rejected (bad JSON): ${command}`);
    const rp = ex.svm_dap_response_ptr();
    const rl = ex.svm_dap_response_len();
    const text = rl ? dec.decode(new Uint8Array(memory.buffer).slice(rp, rp + rl)) : '[]';
    const msgs = JSON.parse(text);
    return {
      response: msgs.find((m) => m.type === 'response'),
      events: msgs.filter((m) => m.type === 'event'),
      all: msgs,
    };
  };

  return { send };
}
