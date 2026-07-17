/* QuickJS — a self-validating JS-engine demo for the LLVM→SVM-IR on-ramp
 * (LLVM.md "Pending work" → the QuickJS target).
 *
 * A minimal embedding of Bellard's QuickJS (2024-01-13): create a runtime +
 * context (the standard intrinsics only — no `quickjs-libc`, so no ambient OS
 * surface), evaluate a fixed JS program, stringify the result, and print it.
 * The program is chosen to exercise the engine's density in one shot:
 *   - recursion + loops               (fib, the accumulator loop)
 *   - closures / host callbacks       (Array.prototype.sort with a comparator)
 *   - the object/GC machinery + JSON  (JSON.stringify of a nested literal)
 *   - string methods                  (String.prototype.toLowerCase)
 *   - float formatting                (Number.prototype.toFixed)
 *
 * Built two ways, diffed byte-for-byte (`demo_quickjs_eval_vs_native`):
 *   - guest:  clang -O2 -emit-llvm  → translate → verify → interp/JIT
 *   - native: cc                    → the oracle
 *
 * The QuickJS sources are fetched-not-vendored (MIT, ~1.9 MB of bitcode once
 * linked); see `build_bitcode.sh` for the exact pipeline the test automates.
 */
#include <stdio.h>
#include <string.h>
#include "quickjs.h"

static const char *PROG =
    "function fib(n){ return n<2 ? n : fib(n-1)+fib(n-2); }\n"
    "var s=0; for (var i=0;i<=20;i++) s += fib(i);\n"
    "var arr=[5,3,8,1,9,2,7]; arr.sort(function(a,b){return a-b;});\n"
    "var str = arr.join(',') + ' | sumfib=' + s + ' | ' + "
    "JSON.stringify({a:1,b:[true,null,'x']});\n"
    "str + ' | ' + 'ABC'.toLowerCase() + ' | ' + (0.1+0.2).toFixed(4);\n";

int main(void) {
    JSRuntime *rt = JS_NewRuntime();
    JSContext *ctx = JS_NewContext(rt);
    JSValue val = JS_Eval(ctx, PROG, strlen(PROG), "<eval>", JS_EVAL_TYPE_GLOBAL);
    if (JS_IsException(val)) {
        JSValue exc = JS_GetException(ctx);
        const char *e = JS_ToCString(ctx, exc);
        printf("EXCEPTION: %s\n", e ? e : "?");
        JS_FreeCString(ctx, e);
        JS_FreeValue(ctx, exc);
    } else {
        const char *s = JS_ToCString(ctx, val);
        printf("%s\n", s ? s : "(null)");
        JS_FreeCString(ctx, s);
    }
    JS_FreeValue(ctx, val);
    JS_FreeContext(ctx);
    JS_FreeRuntime(rt);
    return 0;
}
