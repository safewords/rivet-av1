#!/bin/bash
# parts-native.sh label [src-tree] — native timing build: a copy of the tree (default: the working tree) with a timer around coeffs() and
# reconstruct() in transform_block, av1_decbench printing ms per frame of each. Built into nt/<label>; run
# nt/<label>/target/release/examples/av1_decbench clip.ivf 7 1.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"; label="$1"; src="${2:-$S/rivet-av1}"
export PATH="$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS=3 RUSTUP_TOOLCHAIN=1.100.0
mkdir -p "$S/nt/$label"; rsync -a --delete --exclude target --exclude tests/vectors --exclude .git "$src/" "$S/nt/$label/"
cd "$S/nt/$label" || exit 1
python3 -I - <<'PY'
p='src/decoder/residual.rs'
s=open(p).read()
a="let (eob, dequantised) = self.coeffs_dequantised(plane, start_x, start_y, tx_sz);"
if a not in s: a="let eob = self.coeffs(plane, start_x, start_y, tx_sz);"
assert a in s
s=s.replace(a,"let t0 = std::time::Instant::now();\n            "+a+"\n            crate::TIMERS[0].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n            let t0 = std::time::Instant::now();",1)
i=s.index("crate::TIMERS[0]"); j=s.index("\n            }\n", i)+len("\n            }\n")
s=s[:j]+"            crate::TIMERS[1].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n"+s[j:]
open(p,'w').write(s)
p='src/lib.rs'
s=open(p).read()
s+='''
#[doc(hidden)]
pub static TIMERS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
'''
open(p,'w').write(s)
p='examples/av1_decbench.rs'
s=open(p).read()
a="        best = best.min(start.elapsed().as_secs_f64());"
assert a in s
s=s.replace("    for _ in 0..runs {","    let mut parts = Vec::new();\n    for _ in 0..runs {\n        for t in &av1::TIMERS { t.store(0, std::sync::atomic::Ordering::Relaxed); }",1)
s=s.replace(a,"        let el = start.elapsed().as_secs_f64();\n        parts.push((av1::TIMERS[0].load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6 / frames as f64, av1::TIMERS[1].load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6 / frames as f64, el * 1e3 / frames as f64));\n"+a,1)
s=s.replace("    Ok(())\n}","    let med = |f: &dyn Fn(&(f64, f64, f64)) -> f64| { let mut v: Vec<f64> = parts.iter().map(f).collect(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };\n    println!(\"  median of {runs}: frame {:.3} ms, coeffs() {:.3} ms, reconstruct() {:.3} ms\", med(&|p| p.2), med(&|p| p.0), med(&|p| p.1));\n    Ok(())\n}",1)
open(p,'w').write(s)
PY
cargo build --release --example av1_decbench 2>&1 | grep -E "^(error|warning)" -A8 | head -30
ls -la target/release/examples/av1_decbench | awk '{print $5, $9}'
