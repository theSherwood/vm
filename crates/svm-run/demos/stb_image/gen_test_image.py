import zlib, struct

W, H, BPP = 24, 24, 4  # RGBA

# Deterministic pattern: diagonal gradient + checker alpha.
def px(x, y):
    r = (x * 11 + y * 3) & 0xff
    g = (y * 13 + 7) & 0xff
    b = ((x ^ y) * 9) & 0xff
    a = 255 if ((x // 3 + y // 3) & 1) == 0 else 128
    return (r, g, b, a)

rows = []
for y in range(H):
    row = bytearray()
    for x in range(W):
        row.extend(px(x, y))
    rows.append(bytes(row))

def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc: return a
    if pb <= pc: return b
    return c

raw = bytearray()
prev = bytes(W * BPP)
for y, row in enumerate(rows):
    ft = y % 5  # cycle None/Sub/Up/Average/Paeth -> exercise every unfilter path
    out = bytearray([ft])
    for i in range(len(row)):
        x = row[i]
        a = row[i - BPP] if i >= BPP else 0
        b = prev[i]
        c = prev[i - BPP] if i >= BPP else 0
        if ft == 0:   f = x
        elif ft == 1: f = (x - a) & 0xff
        elif ft == 2: f = (x - b) & 0xff
        elif ft == 3: f = (x - ((a + b) >> 1)) & 0xff
        else:         f = (x - paeth(a, b, c)) & 0xff
        out.append(f)
    raw.extend(out)
    prev = row

def chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)

ihdr = struct.pack(">IIBBBBB", W, H, 8, 6, 0, 0, 0)  # 8-bit, colortype 6 (RGBA)
idat = zlib.compress(bytes(raw), 9)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")

# Emit the .inc
with open("test_image.inc", "w") as f:
    f.write("/* A %dx%d RGBA PNG, generated deterministically (gen_test_image.py — all five row\n" % (W, H))
    f.write(" * filters cycled to exercise the decoder's None/Sub/Up/Average/Paeth unfilter paths).\n")
    f.write(" * Decoded RGBA output is the differential oracle; do not edit by hand. */\n")
    f.write("static const unsigned char PNG[] = {\n")
    for i in range(0, len(png), 16):
        f.write("  " + ",".join(str(x) for x in png[i:i+16]) + ",\n")
    f.write("};\n")
    f.write("static const unsigned PNG_LEN = %d;\n" % len(png))
    f.write("static const int IMG_W = %d, IMG_H = %d;\n" % (W, H))

print("PNG bytes:", len(png), " raw:", len(raw), " decoded RGBA bytes:", W*H*BPP)
