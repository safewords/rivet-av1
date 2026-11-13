use std::time::Instant;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let trace = std::fs::read(&args[1]).unwrap();
    let rounds: usize = args.get(2).map_or(9, |s| s.parse().unwrap());
    let w = coefbench::common::load(&trace);
    let mut st = coefbench::common::St::new();
    let nv = coefbench::NAMES.len();
    let nz = w.coefs.iter().filter(|&&c| c != 0).count();
    println!("{} blocks, {} coefficients ({} not zero), {} bytes", w.blocks.len(), w.coefs.len(), nz, w.data.len());
    for v in 0..nv {
        assert!(coefbench::verify(&w, &mut st, v as u32), "variant {v} decodes something else");
    }
    let mut t = vec![Vec::new(); nv];
    for _ in 0..rounds {
        for v in 0..nv {
            let t0 = Instant::now();
            let s = coefbench::run_variant(&w, &mut st, v as u32);
            let dt = t0.elapsed().as_secs_f64();
            assert_eq!(s, w.sum);
            t[v].push(dt * 1e9 / w.coefs.len() as f64);
        }
    }
    for v in 0..nv {
        t[v].sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{:76} min {:6.2} ns/coef  median {:6.2}", coefbench::NAMES[v], t[v][0], t[v][rounds / 2]);
    }
}
