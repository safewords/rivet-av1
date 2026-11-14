//! The structure of a group of frames: the order frames are coded in,
//! which of them are hidden, and each frame's layer of the pyramid.
//!
//! A group is the `n` frames after an anchor (the last frame shown: a key
//! frame or the previous group's alt-ref frame), numbered 1 to `n` in
//! display order. The last of them, the alt-ref frame, is coded first and
//! not shown (layer 0). Then, recursively, the frame in the middle of each
//! stretch still to code is coded hidden one layer up, until a stretch is
//! short enough that its frames are coded in display order and shown as
//! they are coded; each hidden frame is shown, with `show_existing_frame`,
//! once every frame before it has been. For eight frames:
//!
//! ```text
//! coded:  8h 4h 2h 1 [2] 3 [4] 6h 5 [6] 7 [8]
//! layer:  0  1  2  3     3     2  3     3
//! ```
//!
//! (`h` hidden, `[i]` frame `i` shown again). Every frame between two
//! coded ones is thus predicted from both sides, from the nearest frames of
//! the layers below it, which are coded finer the lower their layer (each
//! is a reference for more frames than the one above it).

/// One step of a group's coding order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// Code frame `idx` (1-based in the group) at `layer`; shown at once,
    /// or hidden until a `Show`.
    Code { idx: u32, layer: u8, show: bool },
    /// Show the hidden frame `idx` (`show_existing_frame`).
    Show { idx: u32 },
}

/// The coding order of a group of `n` frames; `n` of 1 or 2 is coded in
/// display order with no hidden frame.
pub(crate) fn plan(n: u32) -> Vec<Step> {
    let mut out = Vec::new();
    if n <= 2 {
        for idx in 1..=n {
            out.push(Step::Code {
                idx,
                layer: 1,
                show: true,
            });
        }
        return out;
    }
    out.push(Step::Code {
        idx: n,
        layer: 0,
        show: false,
    });
    stretch(&mut out, 0, n, 1);
    out
}

/// Frames `lo + 1 .. hi` (frame `lo` shown, `hi` coded and hidden), then
/// `hi` shown.
fn stretch(out: &mut Vec<Step>, lo: u32, hi: u32, layer: u8) {
    let between = hi - lo - 1;
    if between <= 2 {
        for idx in lo + 1..hi {
            out.push(Step::Code {
                idx,
                layer,
                show: true,
            });
        }
        out.push(Step::Show { idx: hi });
    } else {
        let mid = (lo + hi).div_ceil(2);
        out.push(Step::Code {
            idx: mid,
            layer,
            show: false,
        });
        stretch(out, lo, mid, layer + 1);
        // (Which ends showing `hi`.)
        stretch(out, mid, hi, layer + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each frame is coded once and shown once, in display order, and
    /// every hidden frame is coded before it is shown.
    #[test]
    fn every_frame_once_in_order() {
        for n in 1..=40 {
            let steps = plan(n);
            let mut coded = vec![false; n as usize + 1];
            let mut shown = Vec::new();
            for s in steps {
                match s {
                    Step::Code { idx, show, .. } => {
                        assert!(!coded[idx as usize], "{n}: {idx} coded twice");
                        coded[idx as usize] = true;
                        if show {
                            shown.push(idx);
                        }
                    }
                    Step::Show { idx } => {
                        assert!(coded[idx as usize], "{n}: {idx} shown before coded");
                        shown.push(idx);
                    }
                }
            }
            assert_eq!(shown, (1..=n).collect::<Vec<_>>(), "{n}");
        }
    }

    #[test]
    fn eight() {
        let order: Vec<(u32, u8, bool)> = plan(8)
            .into_iter()
            .filter_map(|s| match s {
                Step::Code { idx, layer, show } => Some((idx, layer, show)),
                Step::Show { .. } => None,
            })
            .collect();
        assert_eq!(
            order,
            [
                (8, 0, false),
                (4, 1, false),
                (2, 2, false),
                (1, 3, true),
                (3, 3, true),
                (6, 2, false),
                (5, 3, true),
                (7, 3, true)
            ]
        );
    }
}
