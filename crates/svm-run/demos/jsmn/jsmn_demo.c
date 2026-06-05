#define JSMN_STATIC
#include "jsmn.h"

int write(int fd, char *buf, long n);
static void puts_(char *s) { long n = 0; while (s[n]) n++; write(1, s, n); }
static void putn(char *s, int n) { write(1, s, n); }
static void putint(long v) {
  char b[24]; int i = 0;
  if (v < 0) { char m = '-'; write(1, &m, 1); v = -v; }
  if (v == 0) { char z = '0'; write(1, &z, 1); return; }
  while (v) { b[i++] = (char)('0' + v % 10); v /= 10; }
  while (i) { char c = b[--i]; write(1, &c, 1); }
}
static char *type_name(int t) {
  if (t == JSMN_OBJECT) return "object";
  if (t == JSMN_ARRAY) return "array";
  if (t == JSMN_STRING) return "string";
  if (t == JSMN_PRIMITIVE) return "primitive";
  return "undefined";
}

int main(void) {
  char *json = "{\"name\":\"svm\",\"nums\":[1,2,3],\"ok\":true,\"nested\":{\"x\":42}}";
  long len = 0; while (json[len]) len++;
  jsmn_parser p; jsmn_init(&p);
  static jsmntok_t toks[128];
  int n = jsmn_parse(&p, json, (size_t)len, toks, 128);
  if (n < 0) { puts_("parse error "); putint(n); puts_("\n"); return 1; }
  putint(n); puts_(" tokens\n");
  for (int i = 0; i < n; i++) {
    jsmntok_t *t = &toks[i];
    puts_("  "); puts_(type_name(t->type));
    puts_(" size="); putint(t->size);
    puts_(" \""); putn(json + t->start, t->end - t->start); puts_("\"\n");
  }
  return 0;
}
