#define SVM_GUEST 1
#include "os_shim.c"
#include "libc_shim.c"
#include "locale_shim.c"
#include "time_shim.c"
#include "proc_shim.c"
#include "ipc_shim.c"
#include "stdio_shim.c"
#include "printf_shim.c"
#include "../strtod/strtod.c" /* real correctly-rounded strtod (the on-ramp's is a trap stub) */
#include "scanf_shim.c"
