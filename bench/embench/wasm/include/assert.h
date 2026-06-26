/* Minimal freestanding-wasm `<assert.h>` shim (see string.h header comment). The Embench build always
 * passes `-DNDEBUG`, so `assert` is a no-op anyway; this just avoids the missing-header error. */
#ifndef _EMBENCH_WASM_ASSERT_H
#define _EMBENCH_WASM_ASSERT_H
#define assert(x) ((void)0)
#endif
