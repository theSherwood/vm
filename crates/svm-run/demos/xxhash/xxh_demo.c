#define XXH_INLINE_ALL
#define XXH_NO_XXH3
#define XXH_NO_STREAM
#define XXH_VECTOR XXH_SCALAR
#include <stddef.h>
/* whole-program build, no libc: provide the two mem* functions xxHash uses. */
void *memcpy(void *d, const void *s, size_t n) {
  unsigned char *dp = d; const unsigned char *sp = s;
  for (size_t i = 0; i < n; i++) dp[i] = sp[i];
  return d;
}
void *memset(void *s, int c, size_t n) {
  unsigned char *p = s;
  for (size_t i = 0; i < n; i++) p[i] = (unsigned char)c;
  return s;
}
#include "xxhash.h"

int write(int fd, char *buf, long n);
static void puts_(char *s){ long n=0; while(s[n])n++; write(1,s,n); }
static void puthex(unsigned long v, int nbytes){
  char *hx="0123456789abcdef"; char out[16]; int i=0;
  for(int b=nbytes-1;b>=0;b--){ unsigned x=(unsigned)((v>>(b*8))&0xff); out[i++]=hx[x>>4]; out[i++]=hx[x&15]; }
  write(1,out,i);
}
static int slen(const char*s){int n=0;while(s[n])n++;return n;}
static void hash(const char *msg){
  XXH32_hash_t h32 = XXH32(msg,(size_t)slen(msg),0);
  XXH64_hash_t h64 = XXH64(msg,(size_t)slen(msg),0);
  puts_("XXH32="); puthex((unsigned long)h32,4);
  puts_(" XXH64="); puthex((unsigned long)h64,8); puts_("\n");
}
int main(void){ hash(""); hash("abc"); hash("The quick brown fox jumps over the lazy dog"); return 0; }
