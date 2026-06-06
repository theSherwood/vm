# tinfl (DEFLATE/zlib inflate) demo

`miniz_tinfl.c`, `miniz_tinfl.h`, `miniz_common.h`, and `miniz_export.h` are from
[miniz](https://github.com/richgel999/miniz) by Rich Geldreich et al. (**MIT** — see the
license block atop `miniz_tinfl.c`). `tinfl` is miniz's standalone DEFLATE/zlib *inflate*
engine. The files are vendored unmodified except for one line in `miniz_tinfl.c`: its
`#include "miniz.h"` (the full miniz umbrella, which also pulls in the deflate/zip headers we
don't ship) is changed to `#include "miniz_tinfl.h"` so the inflate path is self-contained.

`tinfl_demo.c` builds it for the libc-free sandbox — `MINIZ_NO_STDIO` / `MINIZ_NO_TIME` /
`MINIZ_NO_MALLOC` / `NDEBUG`, plus the `memcpy`/`memset` it needs — and inflates `blob.inc`,
writing the result to stdout. `svm-run`'s output must match a native `cc` build byte-for-byte.

```sh
cargo run -p svm-run -- crates/svm-run/demos/tinfl/tinfl_demo.c
```

`blob.inc` is a zlib stream of the line `"The quick brown fox jumps over the lazy dog.\n"`
repeated six times (270 bytes, compressed to 55), generated with stock zlib:

```python
import zlib
data = b"The quick brown fox jumps over the lazy dog.\n" * 6
z = zlib.compress(data, 9)
print("static const unsigned char BLOB[] = {%s};" % ",".join(map(str, z)))
print("static const unsigned BLOB_LEN = %d;" % len(z))
print("static const unsigned ORIG_LEN = %d;" % len(data))
```

A new shape for the shakedown series: a coroutine-style inflate state machine (a deeply nested
`switch` driven by miniz's `TINFL_CR_*` macros), bit-buffer shifts, Huffman fast/slow lookup
tables, and a 32 KiB LZ77 dictionary carried inside `tinfl_decompressor` — a good stress test of
goto/switch lowering and struct layout. It ran identically to a native build with no new fixes.
