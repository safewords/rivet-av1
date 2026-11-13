//! Variants of the symbol decoder, every one derived from this crate's own src/symbol.rs, decoding the symbols
//! `record.sh` recorded from a real decode: what steps 8 and 9 of docs/wasm-performance.md were chosen by.
#![allow(clippy::all)]
pub mod enc;
pub mod v0;
pub mod v1;
pub mod v3;
pub mod v8;
pub mod v9;
pub mod v10;

pub trait SymDec<'a> {
    fn new(data: &'a [u8]) -> Self;
    fn sym(&mut self, cdf: &mut [u16]) -> usize;
    fn sym2(&mut self, cdf: &mut [u16; 3]) -> usize;
    fn sym3(&mut self, cdf: &mut [u16; 4]) -> usize;
    fn sym4(&mut self, cdf: &mut [u16; 5]) -> usize;
    fn bit(&mut self) -> u32;
}

pub struct Work {
    pub ops: Vec<u32>, // id << 8 | n << 3 ... : (id << 10) | (n << 5) | sym
    pub cdfs: Vec<[u16; 18]>,
    pub data: Vec<u8>,
    pub sum: u64,
}

pub fn load(trace: &[u8]) -> Work {
    load_filtered(trace, FILTER)
}
/// FILTER: 0 all, 4 only the 4-symbol reads, 1 only the bools, 2 the 4-symbol reads and the bools.
pub const FILTER: u32 = match option_env!("FILTER") { Some(s) => s.as_bytes()[0] as u32 - 48, None => 0 };
pub fn load_filtered(trace: &[u8], filter: u32) -> Work {
    let mut ops = Vec::new();
    let mut cdfs: Vec<[u16; 18]> = Vec::new();
    let mut i = 0;
    while i < trace.len() {
        match trace[i] {
            0 => {
                if filter == 0 || filter == 1 || filter == 2 { ops.push(trace[i + 1] as u32); }
                i += 2;
            }
            t => {
                let id = u32::from_le_bytes(trace[i + 1..i + 5].try_into().unwrap());
                let n = trace[i + 5] as u32;
                let s = trace[i + 6] as u32;
                i += 7;
                if t == 2 {
                    let mut c = [0u16; 18];
                    for k in 0..=n as usize {
                        c[k] = u16::from_le_bytes(trace[i..i + 2].try_into().unwrap());
                        i += 2;
                    }
                    assert_eq!(id as usize, cdfs.len());
                    cdfs.push(c);
                }
                if filter == 0 || ((filter == 4 || filter == 2) && n == 4) { ops.push(id << 10 | n << 5 | s); }
            }
        }
    }
    // Encode with the crate's own encoder.
    let mut e = enc::SymbolEncoder::new();
    let mut c = cdfs.clone();
    let mut sum = 0u64;
    for &op in &ops {
        let n = (op >> 5) & 31;
        let s = op & 31;
        sum += s as u64;
        if n == 0 {
            e.write_bool(s);
        } else {
            e.write_symbol(&mut c[(op >> 10) as usize][..n as usize + 1], s as usize);
        }
    }
    let data = e.finish();
    Work { ops, cdfs, data, sum }
}

#[inline(never)]
pub fn run<'a, D: SymDec<'a>>(w: &'a Work, cdfs: &mut [[u16; 18]]) -> u64 {
    cdfs.copy_from_slice(&w.cdfs);
    let mut d = D::new(&w.data);
    let mut sum = 0u64;
    for &op in &w.ops {
        let n = (op >> 5) & 31;
        let s = if n == 0 {
            d.bit() as usize
        } else {
            let c = &mut cdfs[(op >> 10) as usize];
            match n {
                4 => d.sym4((&mut c[..5]).try_into().unwrap()),
                3 => d.sym3((&mut c[..4]).try_into().unwrap()),
                2 => d.sym2((&mut c[..3]).try_into().unwrap()),
                _ => d.sym(&mut c[..n as usize + 1]),
            }
        };
        sum += s as u64;
    }
    sum
}

pub fn run_variant(w: &Work, cdfs: &mut [[u16; 18]], v: u32) -> u64 {
    match v {
        0 => run::<v0::D>(w, cdfs),
        1 => run::<v1::D>(w, cdfs),
        2 => run::<v3::D<2, 1, true>>(w, cdfs),
        3 => run::<v8::D<0, 0, false>>(w, cdfs),
        4 => run::<v8::D<0, 1, false>>(w, cdfs),
        5 => run::<v8::D<0, 3, false>>(w, cdfs),
        6 => run::<v8::D<2, 1, false>>(w, cdfs),
        7 => run::<v8::D<2, 3, false>>(w, cdfs),
        8 => run::<v8::D<2, 3, true>>(w, cdfs),
        9 => run::<v8::D<0, 3, true>>(w, cdfs),
        10 => run::<v9::D<false>>(w, cdfs),
        11 => run::<v9::D<true>>(w, cdfs),
        12 => run::<v10::D>(w, cdfs),
        13 => run::<v3::D<0, 2, true>>(w, cdfs),
        _ => 0,
    }
}
pub const NAMES: [&str; 14] = [
    "0 before step 8: bits from three bytes at each renormalisation",
    "1 step 8: the 64-bit window",
    "2 window, search by selects, adaptation by ifs",
    "3 window in registers",
    "4 registers, adaptation by ifs",
    "5 registers, adaptation by masks from the symbol",
    "6 registers, search by selects, adaptation by ifs",
    "7 registers, search by selects, adaptation by masks",
    "8 = 7 with the bit by selects",
    "9 = 5 with the bit by selects",
    "10 registers, one arm for each symbol",
    "11 = 10 with the bit by selects",
    "12 step 9: registers, no branch on the data",
    "13 window, adaptation of four entries as one vector",
];

// ---- wasm entry points ----
static mut WORK: Option<(Work, Vec<[u16; 18]>)> = None;
#[cfg(target_arch = "wasm32")]
static TRACE: &[u8] = include_bytes!(env!("TRACE"));
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn init() -> u32 {
    let w = load(TRACE);
    let c = w.cdfs.clone();
    let n = w.ops.len() as u32;
    unsafe { WORK = Some((w, c)) };
    n
}
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn bench(v: u32) -> u32 {
    #[allow(static_mut_refs)]
    let (w, c) = unsafe { WORK.as_mut().unwrap() };
    let s = run_variant(w, c, v);
    (s == w.sum) as u32
}
