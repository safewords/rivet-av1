#!/bin/bash
# record.sh clip.ivf [out-prefix] [frames] — records what the microbenchmarks replay, from a patched copy of the tree decoding
# the clip with CoefficientReader::Reference (every symbol then goes through SymbolDecoder::read_symbol or read_bool):
# <out>-symbols.bin, every symbol read (which CDF, its size, the symbol, and the CDF as it was when first seen; a bit for
# read_bool), and <out>-blocks.bin, every transform block with a coefficient (size, class, plane type, eob and the
# coefficients in scan order). The first `frames` packets (default 12); <out> defaults to $WASM_PERF/trace.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"
export PATH="$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS=3 RUSTUP_TOOLCHAIN=1.100.0
mkdir -p "$S/record-tree"; rsync -a --delete --exclude target --exclude tests/vectors --exclude .git "$S/rivet-av1/" "$S/record-tree/"
cd "$S/record-tree" || exit 1
python3 -I - <<'PY'
p='src/symbol.rs'
s=open(p).read()
a="    pub(crate) fn read_symbol(&mut self, cdf: &mut [u16]) -> usize {\n        self.read_symbol_inline(cdf)\n    }\n"
assert a in s
s=s.replace(a,'''    pub(crate) fn read_symbol(&mut self, cdf: &mut [u16]) -> usize {
        let before = cdf.to_vec();
        let symbol = self.read_symbol_inline(cdf);
        RECORD.with(|t| {
            let mut t = t.borrow_mut();
            if t.on {
                let next = t.ids.len() as u32;
                let id = *t.ids.entry(cdf.as_ptr() as usize).or_insert(next);
                t.symbols.push(if id == next { 2 } else { 1 });
                t.symbols.extend_from_slice(&id.to_le_bytes());
                t.symbols.push(cdf.len() as u8 - 1);
                t.symbols.push(symbol as u8);
                if id == next {
                    for c in &before {
                        t.symbols.extend_from_slice(&c.to_le_bytes());
                    }
                }
            }
        });
        symbol
    }
''',1)
i=s.index("    pub(crate) fn read_bool(&mut self) -> u32 {")
a="        self.renormalize();\n        bit\n    }\n"
j=s.index(a,i)
s=s[:j]+'''        self.renormalize();
        RECORD.with(|t| {
            let mut t = t.borrow_mut();
            if t.on {
                t.symbols.push(0);
                t.symbols.push(bit as u8);
            }
        });
        bit
    }
'''+s[j+len(a):]
s+='''
#[derive(Default)]
pub struct Record {
    pub on: bool,
    ids: std::collections::HashMap<usize, u32>,
    pub symbols: Vec<u8>,
    pub blocks: Vec<u8>,
}
thread_local! { pub static RECORD: std::cell::RefCell<Record> = std::cell::RefCell::new(Record::default()); }
'''
open(p,'w').write(s)
p='src/decoder/residual.rs'
s=open(p).read()
a="            return (self.coeffs_shared(plane, start_x, start_y, tx_sz), None);\n"
assert a in s
s=s.replace(a,'''            let eob = self.coeffs_shared(plane, start_x, start_y, tx_sz);
            crate::symbol::RECORD.with(|t| {
                let mut t = t.borrow_mut();
                if t.on && eob > 0 {
                    let scan = scan_for(tx_sz, self.plane_tx_type);
                    t.blocks.push(tx_sz as u8);
                    t.blocks.push(get_tx_class(self.plane_tx_type) as u8);
                    t.blocks.push((plane > 0) as u8);
                    t.blocks.extend_from_slice(&(eob as u16).to_le_bytes());
                    for &pos in &scan[..eob] {
                        t.blocks.extend_from_slice(&self.quant[pos as usize].to_le_bytes());
                    }
                }
            });
            return (eob, None);
''',1)
open(p,'w').write(s)
p='src/lib.rs'
s=open(p).read()
s+='''
#[doc(hidden)]
pub fn record(on: bool) -> (Vec<u8>, Vec<u8>) {
    symbol::RECORD.with(|t| {
        let mut t = t.borrow_mut();
        t.on = on;
        (std::mem::take(&mut t.symbols), std::mem::take(&mut t.blocks))
    })
}
'''
open(p,'w').write(s)
p='examples/av1_decbench.rs'
s=open(p).read()
a="        for p in &packets {\n"
assert a in s
s=s.replace("        let mut dec = av1::Decoder::new();\n","        let mut dec = av1::Decoder::new();\n        dec.set_coefficient_reader(av1::decoder::CoefficientReader::Reference);\n        let frames_to_record: usize = std::env::var(\"RECORD_FRAMES\").map_or(12, |s| s.parse().unwrap());\n",1)
s=s.replace(a,"        for (pi, p) in packets.iter().enumerate() {\n            if pi == frames_to_record {\n                let (symbols, blocks) = av1::record(false);\n                let out = std::env::var(\"RECORD_OUT\").unwrap();\n                std::fs::write(format!(\"{out}-symbols.bin\"), symbols)?;\n                std::fs::write(format!(\"{out}-blocks.bin\"), blocks)?;\n            } else if pi == 0 {\n                av1::record(true);\n            }\n",1)
open(p,'w').write(s)
PY
cargo build --release --example av1_decbench 2>&1 | grep -E "^error" -A12
RECORD_OUT="${2:-$S/trace}" RECORD_FRAMES="${3:-12}" ./target/release/examples/av1_decbench "$1" 1 1 | head -1
ls -la "${2:-$S/trace}"-symbols.bin "${2:-$S/trace}"-blocks.bin
