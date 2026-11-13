//! Replays the level and sign passes of the decode-only coefficient reader on the transform blocks `record.sh`
//! recorded from a real decode, to compare variants of the passes with each other, natively and in wasm: what
//! steps 9 and 10 of docs/wasm-performance.md were chosen by.
#![allow(dead_code, clippy::all, unused_imports)]
#[path = "../../../../src/consts.rs"]
mod consts;
#[path = "../../../../src/tables.rs"]
mod tables;
pub mod common;
pub mod sym;
pub mod va;
pub mod vb;
pub mod vc;
pub mod vd;
pub mod ve;
pub mod vf;
pub mod vg;

use common::*;

pub fn run_variant(w: &Work, st: &mut St, v: u32) -> u64 {
    st.cdfs = Cdfs::new();
    let mut d = sym::Dec::new(&w.data);
    let mut sum = 0u64;
    match v {
        0 => for b in &w.blocks { sum += va::read(&mut d, st, b) as u64 },
        1 => for b in &w.blocks { sum += vb::read(&mut d, st, b) as u64 },
        2 => for b in &w.blocks { sum += vc::read(&mut d, st, b) as u64 },
        3 => for b in &w.blocks { sum += vd::read(&mut d, st, b) as u64 },
        4 => for b in &w.blocks { sum += ve::read(&mut d, st, b) as u64 },
        5 => for b in &w.blocks { sum += vf::read(&mut d, st, b) as u64 },
        6 => for b in &w.blocks { sum += vg::read(&mut d, st, b) as u64 },
        _ => {}
    }
    sum
}
pub const NAMES: [&str; 7] = [
    "a: the passes of step 8 (padded map of levels, searching reads)",
    "b: a + scan map, decoder in locals, a bit for each non-zero",
    "c: b + one arm for each symbol in the reads",
    "d: c + the bit by selects",
    "e: b + reads with no branch on the data (step 9)",
    "f: e with the branching bit",
    "g: e + Min3 map, CDF index in the scan map, coeff_br out of line (step 10)",
];

/// Decodes every block with variant `v` and compares the coefficients with the recorded ones.
pub fn verify(w: &Work, st: &mut St, v: u32) -> bool {
    st.cdfs = Cdfs::new();
    let mut d = sym::Dec::new(&w.data);
    for b in &w.blocks {
        match v {
            0 => va::read(&mut d, st, b),
            1 => vb::read(&mut d, st, b),
            2 => vc::read(&mut d, st, b),
            3 => vd::read(&mut d, st, b),
            4 => ve::read(&mut d, st, b),
            5 => vf::read(&mut d, st, b),
            6 => vg::read(&mut d, st, b),
            _ => 0,
        };
        let scan = scan_of(b.tx_sz as usize, b.tx_class as usize);
        let want = &w.coefs[b.start as usize..][..b.eob as usize];
        let mut nz = 0;
        for (c, &x) in want.iter().enumerate() {
            if st.quant[scan[c] as usize] != x {
                return false;
            }
            nz += (x != 0) as usize;
        }
        if st.quant[..seg_eob(b.tx_sz as usize)].iter().filter(|&&x| x != 0).count() != nz {
            return false;
        }
    }
    true
}

// ---- wasm entry points ----
static mut WORK: Option<(Work, Box<St>)> = None;
#[cfg(target_arch = "wasm32")]
static TRACE: &[u8] = include_bytes!(env!("BLOCKS"));
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn init() -> u32 {
    let w = load(TRACE);
    let n = w.coefs.len() as u32;
    unsafe { WORK = Some((w, St::new())) };
    n
}
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn check(v: u32) -> u32 {
    #[allow(static_mut_refs)]
    let (w, st) = unsafe { WORK.as_mut().unwrap() };
    verify(w, st, v) as u32
}
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn bench(v: u32) -> u32 {
    #[allow(static_mut_refs)]
    let (w, st) = unsafe { WORK.as_mut().unwrap() };
    (run_variant(w, st, v) == w.sum) as u32
}
