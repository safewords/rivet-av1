//! Malformed input: arbitrary bytes, and real streams with bits flipped,
//! bytes cut and garbage spliced in, must give errors or frames — never a
//! panic.

mod common;

use proptest::prelude::*;

fn streams() -> Vec<Vec<Vec<u8>>> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut v: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "mkv" || e == "ivf"))
        .collect();
    v.sort();
    v.iter().map(|p| common::packets(p)).collect()
}

fn decode_all(packets: &[Vec<u8>]) {
    let mut d = av1::Decoder::new();
    d.set_coefficient_reader(av1::decoder::CoefficientReader::Checked);
    d.set_max_pixels(1 << 20);
    for p in packets {
        let _ = d.decode(p);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 96, ..ProptestConfig::default() })]

    #[test]
    fn arbitrary_bytes(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
        let mut d = av1::Decoder::new();
        d.set_coefficient_reader(av1::decoder::CoefficientReader::Checked);
        d.set_max_pixels(1 << 20);
        let _ = d.decode(&data);
        // Also as the second frame of a stream, after a valid key frame.
        let s = &streams()[0];
        let mut d = av1::Decoder::new();
        d.set_coefficient_reader(av1::decoder::CoefficientReader::Checked);
        d.set_max_pixels(1 << 20);
        let _ = d.decode(&s[0]);
        let _ = d.decode(&data);
    }

    #[test]
    fn mutated_streams(which in 0usize..12, flips in proptest::collection::vec((any::<usize>(), 0u8..8), 1..8),
                       cut in any::<usize>(), cut_packet in any::<usize>(), splice in proptest::collection::vec(any::<u8>(), 0..64)) {
        let all = streams();
        let mut s = all[which % all.len()].clone();
        let n = s.len();
        for (pos, bit) in flips {
            let p = &mut s[pos % n];
            if !p.is_empty() {
                let i = (pos / n) % p.len();
                p[i] ^= 1 << bit;
            }
        }
        let k = cut_packet % n;
        let len = s[k].len();
        s[k].truncate(cut % (len + 1));
        let k2 = (cut_packet / 7) % n;
        let at = cut % (s[k2].len() + 1);
        s[k2].splice(at..at, splice);
        decode_all(&s);
    }
}
