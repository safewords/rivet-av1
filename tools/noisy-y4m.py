#!/usr/bin/env python3
"""A synthetic noisy clip for `examples/av1_ratetest.rs`: small frames of a
slowly drifting pattern (gradients and a few hard edges) under fresh noise
in every frame, the case where rate control has least to go on before the
first group is coded (the noise costs far more than the picture, and is
new in every frame).

    tools/noisy-y4m.py [--size 160x96] [--frames 40] [--noise 10] out.y4m

Deterministic: the same arguments write the same bytes.
"""

import argparse
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="160x96")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--noise", type=int, default=10, help="the noise's amplitude (8-bit)")
    ap.add_argument("out")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    cw, ch = (w + 1) // 2, (h + 1) // 2
    state = 0x2545F4914F6CDD1D

    def rand():
        # xorshift64: a fixed stream, whatever Python's random does.
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 7
        state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
        return state

    def plane(pw, ph, t, base, scale):
        out = bytearray(pw * ph)
        for y in range(ph):
            for x in range(pw):
                u, v = (x + t * 0.7) / pw, (y + t * 0.3) / ph
                s = base + scale * (0.5 * math.sin(6.0 * u) + 0.3 * math.cos(4.0 * v))
                if (int(u * 5) + int(v * 3)) % 2:
                    s += scale * 0.6
                n = (rand() >> 40) % (2 * a.noise + 1) - a.noise
                out[y * pw + x] = max(0, min(255, int(s + n)))
        return out

    with open(a.out, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F{a.fps}:1 Ip A1:1 C420jpeg\n".encode())
        for t in range(a.frames):
            f.write(b"FRAME\n")
            f.write(plane(w, h, t, 120, 60))
            f.write(plane(cw, ch, t, 128, 20))
            f.write(plane(cw, ch, t, 128, 15))


if __name__ == "__main__":
    main()
