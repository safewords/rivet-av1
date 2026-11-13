use std::time::Instant;
fn main() {
    let cases = itxbench::check();
    assert!(cases != 0, "the transform differs from the scalar reference");
    println!("{cases} cases agree with the scalar reference");
    if std::env::args().nth(1).as_deref() == Some("check") { return; }
    let names = ["4x4", "8x8", "16x16", "32x32", "64x64", "8x16", "16x32", "32x16"];
    for (si, name) in names.iter().enumerate() {
        let mut line = format!("{name:>6}");
        for (full, bounded) in [(true, false), (true, true), (false, false), (false, true)] {
            let iters = 20000;
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t0 = Instant::now();
                std::hint::black_box(itxbench::run(si, full, bounded, iters));
                best = best.min(t0.elapsed().as_secs_f64() * 1e9 / iters as f64);
            }
            line += &format!("  {best:8.0}");
        }
        println!("{line}");
    }
}
