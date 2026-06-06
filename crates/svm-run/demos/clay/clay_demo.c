#define CLAY_IMPLEMENTATION
#define CLAY_DISABLE_SIMD
#include "clay.h"

int write(int fd, char *buf, long n);
static void puts_(char *s){ long n=0; while(s[n]) n++; write(1,s,n); }
static void putint(long v){ char b[24]; int i=0; if(v<0){char m='-';write(1,&m,1);v=-v;}
  if(v==0){char z='0';write(1,&z,1);return;} while(v){b[i++]=(char)('0'+v%10);v/=10;} while(i){char c=b[--i];write(1,&c,1);} }

static Clay_Dimensions measure(Clay_StringSlice text, Clay_TextElementConfig *cfg, void *u){
  (void)cfg; (void)u;
  Clay_Dimensions d; d.width = (float)text.length * 8.0f; d.height = 18.0f; return d;
}
static void onError(Clay_ErrorData e){ (void)e; }

static unsigned char arena[512*1024];

int main(void){
  Clay_SetMaxElementCount(256);
  Clay_Arena a = Clay_CreateArenaWithCapacityAndMemory(sizeof(arena), arena);
  Clay_Initialize(a, (Clay_Dimensions){800,600}, (Clay_ErrorHandler){ onError, 0 });
  Clay_SetMeasureTextFunction(measure, 0);

  Clay_BeginLayout();
  CLAY(CLAY_ID("Outer"), { .layout = { .sizing = { CLAY_SIZING_FIXED(800), CLAY_SIZING_FIXED(600) },
                                       .padding = CLAY_PADDING_ALL(16), .childGap = 8,
                                       .layoutDirection = CLAY_TOP_TO_BOTTOM } }) {
    CLAY(CLAY_ID("Title"), { .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(40) } },
                             .backgroundColor = {80,80,160,255} }) {
      CLAY_TEXT(CLAY_STRING("Hello, Clay on SVM!"), CLAY_TEXT_CONFIG({ .fontSize = 18, .textColor = {255,255,255,255} }));
    }
    CLAY(CLAY_ID("Body"), { .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) } },
                            .backgroundColor = {40,40,40,255} }) {}
  }
  Clay_RenderCommandArray cmds = Clay_EndLayout(0.016f);

  putint(cmds.length); puts_(" render commands:\n");
  for (int i = 0; i < cmds.length; i++) {
    Clay_RenderCommand *c = &cmds.internalArray[i];
    puts_("  cmd "); putint(c->commandType);
    puts_(" bbox=("); putint((long)c->boundingBox.x); puts_(",");
    putint((long)c->boundingBox.y); puts_(" "); putint((long)c->boundingBox.width);
    puts_("x"); putint((long)c->boundingBox.height); puts_(")\n");
  }
  return 0;
}
