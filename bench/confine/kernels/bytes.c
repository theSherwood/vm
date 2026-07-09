// picojpeg-shaped serial byte-stream decode: byte load + data-dependent branch off an unbounded base.
#define B 4096
static unsigned char buf[B];
long run(long n){ for(int i=0;i<B;i++) buf[i]=(unsigned char)((i*7+3)&0xff); unsigned st=1; for(long k=0;k<n;k++){ unsigned c=buf[(unsigned)k&(B-1)]; st=(c&0x80)?(st*33+c):(st+c); } return (long)(int)st; }
