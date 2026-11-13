use std::time::Instant;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let trace = std::fs::read(&args[1]).unwrap();
    let rounds: usize = args.get(2).map_or(9, |s| s.parse().unwrap());
    let w = symbench::load(&trace);
    let mut c = w.cdfs.clone();
    let nv = symbench::NAMES.len();
    let nb = w.ops.iter().filter(|&&o| (o >> 5) & 31 == 0).count();
    let n4 = w.ops.iter().filter(|&&o| (o >> 5) & 31 == 4).count();
    println!("{} ops ({} bools, {} 4-symbol), {} cdfs, {} bytes", w.ops.len(), nb, n4, w.cdfs.len(), w.data.len());
    let mut t = vec![Vec::new(); nv];
    for _ in 0..rounds {
        for v in 0..nv {
            let t0 = Instant::now();
            let s = symbench::run_variant(&w, &mut c, v as u32);
            let dt = t0.elapsed().as_secs_f64();
            assert_eq!(s, w.sum, "variant {v} decodes something else");
            t[v].push(dt * 1e9 / w.ops.len() as f64);
        }
    }
    for v in 0..nv {
        t[v].sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{:66} min {:6.2} ns/op  median {:6.2}", symbench::NAMES[v], t[v][0], t[v][rounds / 2]);
    }
}
