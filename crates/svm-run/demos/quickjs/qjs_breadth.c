/* QuickJS breadth demo — exercises a wide slice of the JavaScript language in one program, to prove
 * the on-ramp runs far more than the `qjs_eval.c` smoke test: regular expressions (`libregexp`),
 * exceptions (`try`/`catch`), generators, `Map`/`Set`, closures, destructuring + spread, template-ish
 * string building, `JSON.parse`/`stringify` round-trip, `Object`/`Array` higher-order methods, `Date`
 * (deterministic epoch), and integer `Math`. Output is one `|`-joined line, diffed byte-for-byte
 * against native (`demo_quickjs_breadth_vs_native`).
 *
 * NOTE — BigInt is deliberately omitted: `2n ** 64n` / `(7n).toString()` are currently miscompiled
 * through `libbf` (see ISSUES.md / LLVM.md "Active target — QuickJS"), the one known JS-surface gap.
 * Transcendental `Math` (sqrt/sin/…) is also omitted so this demo doesn't depend on the guest-libm
 * link — the breadth here is pure engine + regex + Unicode.
 */
#include <stdio.h>
#include <string.h>
#include "quickjs.h"

static const char *PROG =
    "var o=[];\n"
    "var m='user@example.com'.match(/(\\w+)@(\\w+)\\.(\\w+)/);"
    "o.push('re:'+m[1]+'/'+m[2]+'/'+m[3]+'/'+'a1b2c3'.replace(/\\d/g,'#'));\n"
    "try{null.x;}catch(e){o.push('exc:'+e.constructor.name);}\n"
    "function* g(n){for(let i=0;i<n;i++)yield i*i;} o.push('gen:'+[...g(5)].join(','));\n"
    "var mp=new Map([['a',1],['b',2]]);mp.set('c',3);o.push('map:'+[...mp].map(([k,v])=>k+v).join(','));\n"
    "o.push('set:'+[...new Set([1,2,2,3,3,3])].join(','));\n"
    "var cnt=(()=>{let c=0;return ()=>++c;})();o.push('clo:'+[cnt(),cnt(),cnt()].join(''));\n"
    "var {x,y=10}={x:1};o.push('des:'+x+'/'+y+'/'+[...[1,2,3],4,5].join(''));\n"
    "o.push('str:'+'Hello World'.split(' ').map(w=>w[0]).join('')+'/'+'abc'.padStart(5,'*'));\n"
    "o.push('json:'+JSON.stringify(JSON.parse('{\"n\":{\"a\":[1,2,{\"d\":true}]}}')));\n"
    "var ob={a:1,b:2,c:3};o.push('obj:'+Object.keys(ob).join('')+'/'+Object.values(ob).reduce((a,b)=>a+b,0));\n"
    "o.push('arr:'+[1,2,3,4,5,6].filter(x=>x%2).map(x=>x*x).reduce((a,b)=>a+b,0));\n"
    "o.push('date:'+new Date(0).getUTCFullYear());\n"
    "o.push('math:'+Math.max(3,7,2)+'/'+Math.round(2.5)+'/'+(17>>>1)+'/'+(255&15)+'/'+(6%4));\n"
    "o.join('|');\n";

int main(void) {
    JSRuntime *rt = JS_NewRuntime();
    JSContext *ctx = JS_NewContext(rt);
    JSValue v = JS_Eval(ctx, PROG, strlen(PROG), "<breadth>", JS_EVAL_TYPE_GLOBAL);
    if (JS_IsException(v)) {
        JSValue e = JS_GetException(ctx);
        const char *s = JS_ToCString(ctx, e);
        printf("EXC:%s\n", s ? s : "?");
        JS_FreeCString(ctx, s);
        JS_FreeValue(ctx, e);
    } else {
        const char *s = JS_ToCString(ctx, v);
        printf("%s\n", s ? s : "?");
        JS_FreeCString(ctx, s);
    }
    JS_FreeValue(ctx, v);
    JS_FreeContext(ctx);
    JS_FreeRuntime(rt);
    return 0;
}
