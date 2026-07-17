/* QuickJS interactive editor guest — the browser-playground REPL. Reads a JS program from **stdin**
 * (the Stream.read capability), evaluates it, and prints anything the program `print`s /
 * `console.log`s plus the completion value of the last expression. The playground page pipes the JS
 * editor's text in as stdin, so a user writes and runs JS entirely client-side, in the sandbox.
 *
 * No `quickjs-libc`, so no ambient OS surface — the only capabilities are the powerbox `Stream`
 * (stdin/stdout) the on-ramp binds. Stateless: each Run is a fresh runtime, matching the
 * "run the whole buffer" editor model (as in `sqlite_repl.c`). Built into a `.svmb` playground asset
 * by `browser/build-onramp-assets.mjs`.
 */
#include <stdio.h>
#include <string.h>
#include "quickjs.h"

extern long read(int fd, void *buf, long n);

/* 1 MiB program buffer — the editor's text, read from stdin. */
static char src[1 << 20];

/* `print(...)` / `console.log(...)` — space-separated args, newline-terminated, to stdout. The
 * common QuickJS-example convention; makes the REPL usable without pulling in `quickjs-libc`. */
static JSValue js_print(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    (void)this_val;
    for (int i = 0; i < argc; i++) {
        if (i)
            putchar(' ');
        const char *s = JS_ToCString(ctx, argv[i]);
        if (s) {
            fputs(s, stdout);
            JS_FreeCString(ctx, s);
        }
    }
    putchar('\n');
    return JS_UNDEFINED;
}

int main(void) {
    long len = 0;
    for (;;) {
        long r = read(0, src + len, (long)sizeof(src) - 1 - len);
        if (r <= 0)
            break;
        len += r;
        if (len >= (long)sizeof(src) - 1)
            break;
    }
    src[len] = 0;

    JSRuntime *rt = JS_NewRuntime();
    JSContext *ctx = JS_NewContext(rt);

    /* Bind `print` and a minimal `console.log`. */
    JSValue global = JS_GetGlobalObject(ctx);
    JS_SetPropertyStr(ctx, global, "print", JS_NewCFunction(ctx, js_print, "print", 1));
    JSValue console = JS_NewObject(ctx);
    JS_SetPropertyStr(ctx, console, "log", JS_NewCFunction(ctx, js_print, "log", 1));
    JS_SetPropertyStr(ctx, global, "console", console);
    JS_FreeValue(ctx, global);

    JSValue val = JS_Eval(ctx, src, strlen(src), "<repl>", JS_EVAL_TYPE_GLOBAL);
    if (JS_IsException(val)) {
        JSValue exc = JS_GetException(ctx);
        const char *e = JS_ToCString(ctx, exc);
        printf("Uncaught %s\n", e ? e : "exception");
        JS_FreeCString(ctx, e);
        JS_FreeValue(ctx, exc);
    } else {
        /* Print the completion value (the last expression), REPL-style. */
        const char *s = JS_ToCString(ctx, val);
        printf("%s\n", s ? s : "undefined");
        JS_FreeCString(ctx, s);
    }
    JS_FreeValue(ctx, val);
    JS_FreeContext(ctx);
    JS_FreeRuntime(rt);
    return 0;
}
