/* Corpus shakedown: run Loup Vaillant's Monocypher (BSD-2 / CC0, public domain) in the
 * sandbox. This is the crypto / 64-bit-carry-arithmetic stressor — it extends the integer
 * corpus (SHA-256, xxHash, crc32) into modern crypto primitives whose field arithmetic and
 * constant-time bit-twiddling are brutal on the 64-bit shift/rotate/multiply paths.
 *
 * Each primitive's output is printed as hex via write(), so any divergence between native cc
 * and our JIT shows up directly in the digits (the native build is the byte-exact oracle —
 * no float formatting involved). On top of that, the X25519 section is a real known-answer
 * test: Alice and Bob derive public keys and independently compute the shared secret; the two
 * MUST agree (the ECDH invariant), and the program's exit code is the number of mismatches —
 * so the elliptic-curve field arithmetic is self-validated, not just differenced.
 *
 * monocypher.c is fully freestanding (it includes only its own header — no libc), so the demo
 * needs no synthesized libc beyond the `write` powerbox primitive. */

#include <stddef.h>
#include <stdint.h>

#include "monocypher.c"

int write(int fd, char *buf, long n);

static void puts_(const char *s) {
  int n = 0;
  while (s[n]) n++;
  write(1, (char *)s, n);
}

/* Write `n` bytes of `h` as lowercase hex, then a newline — the corpus digest convention. */
static void puthex(const uint8_t *h, int n) {
  const char *hx = "0123456789abcdef";
  char out[2];
  for (int i = 0; i < n; i++) {
    out[0] = hx[h[i] >> 4];
    out[1] = hx[h[i] & 15];
    write(1, out, 2);
  }
  char nl = '\n';
  write(1, &nl, 1);
}

static int slen(const char *s) {
  int n = 0;
  while (s[n]) n++;
  return n;
}

int main(void) {
  /* --- BLAKE2b: hash a fixed message (default 64-byte digest). --- */
  const char *msg = "The quick brown fox jumps over the lazy dog";
  uint8_t hash[64];
  crypto_blake2b(hash, sizeof(hash), (const uint8_t *)msg, slen(msg));
  puts_("blake2b: ");
  puthex(hash, sizeof(hash));

  /* --- ChaCha20 (djb): encrypt a fixed plaintext with a fixed key/nonce/counter. --- */
  uint8_t key[32];
  uint8_t nonce[8];
  for (int i = 0; i < 32; i++) key[i] = (uint8_t)(i + 1);
  for (int i = 0; i < 8; i++) nonce[i] = (uint8_t)(0x40 + i);
  const char *pt = "Monocypher in the sandbox: ChaCha20 stream cipher KAT.";
  int ptlen = slen(pt);
  uint8_t ct[64];
  crypto_chacha20_djb(ct, (const uint8_t *)pt, (size_t)ptlen, key, nonce, 0);
  puts_("chacha20: ");
  puthex(ct, ptlen);

  /* --- Poly1305: MAC the ciphertext under the same 32-byte key. --- */
  uint8_t mac[16];
  crypto_poly1305(mac, ct, (size_t)ptlen, key);
  puts_("poly1305: ");
  puthex(mac, sizeof(mac));

  /* --- X25519 ECDH known-answer test (the field-arithmetic stressor). --- */
  uint8_t alice_sk[32], bob_sk[32];
  for (int i = 0; i < 32; i++) {
    alice_sk[i] = (uint8_t)(0xa0 + i);
    bob_sk[i] = (uint8_t)(0xb0 + i);
  }
  uint8_t alice_pk[32], bob_pk[32];
  crypto_x25519_public_key(alice_pk, alice_sk);
  crypto_x25519_public_key(bob_pk, bob_sk);

  uint8_t alice_shared[32], bob_shared[32];
  crypto_x25519(alice_shared, alice_sk, bob_pk);
  crypto_x25519(bob_shared, bob_sk, alice_pk);

  puts_("x25519:   ");
  puthex(alice_shared, sizeof(alice_shared));

  /* The ECDH invariant: both sides must derive the identical shared secret. */
  int mismatches = 0;
  for (int i = 0; i < 32; i++)
    if (alice_shared[i] != bob_shared[i]) mismatches++;
  puts_(mismatches == 0 ? "ecdh: ok\n" : "ecdh: FAIL\n");

  return mismatches;
}
