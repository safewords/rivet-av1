//! Average-bitrate rate control.
//!
//! Each frame's quantiser comes from a plan over the next `horizon` frames:
//! the quantiser at which this frame (at its offset: key and golden
//! frames are coded finer) and the inter frames after it are
//! predicted to spend the horizon's share of the budget, plus what earlier
//! frames left unspent or minus what they overspent.
//!
//! The prediction is a model per class, refitted after every frame from
//! what the frame really cost:
//!
//! - key frames: `k_key * pixels * (intra + C0) / qstep^E`;
//! - inter frames: `k_inter * pixels * (inter + C0) / qstep^E`, plus, when
//!   the frame is coded finer than its reference was, the refinement of the
//!   reference's error: the key model's cost at this quantiser less its cost
//!   at the reference's.
//!
//! `intra` and `inter` are measured on the source before coding, on a
//! half-size luma, per 16x16 block: the mean absolute deviation from the
//! block's mean, and the smaller of that and the mean absolute difference
//! from the previous source frame. A scene cut thus raises its frame's
//! predicted cost, and the plan its quantiser, before the frame is coded.
//! Inter quantisers fall by at most `MAX_FALL` a frame, so the plan cannot
//! swing into refining a whole frame at once.
//!
//! With alt-ref groups the plan is made a group at a time, before its
//! first frame is coded, and again after each of its frames: the quantiser
//! at which the frames still to code, each at its layer's offset, are
//! predicted to spend what is left of the group's budget (the group's
//! shares, plus what earlier frames left or minus what they overspent,
//! repaid over the group). A key frame is held until the group after it
//! is, and planned with it at its own offset: a key frame and its group
//! spend their shares between them, so the first group of a clip, or of a
//! segment coded on its own, lands on the rate instead of leaving the
//! groups after it to repay it.
//!
//! A group's complexities come from the lookahead (`lookahead`), measured
//! on its sources before it is coded: per 16x16 block, the SATD of the
//! cheapest of an intra prediction and the predictions from the frames the
//! block's frame will be predicted from in the pyramid (an alt-ref frame
//! across the whole group, a frame of the top layer from its neighbours).
//! Each layer has a model of its own (`k_layer * pixels * (inter + C0) /
//! qstep^E`); until a frame of a layer is coded its model is a guess
//! against the key frame's (`LAYER_PRIOR`), and the first frame of each
//! layer scales the guesses of the layers not yet coded by its error. A
//! frame coded finer than the frames it is predicted from (the alt-ref
//! frame and its anchor, in a static scene most of its cost; a frame
//! planned again finer than the frames before it) carries their refinement
//! term. Planning again after every frame is what holds a group to its
//! budget: the models' first guesses are within about a factor of two,
//! and the frames coded first, the alt-ref frame and the layers below the
//! top, are the dearest. A key frame planned with its group, or the first
//! alt-ref frame, far from its prediction is coded again with the model
//! refitted; a frame that costs far beyond its prediction and what is left
//! of the group's budget (noise it refines that the source did not show)
//! is coded again, coarser.

use crate::tables::AC_QLOOKUP;

/// The exponent of the quantiser step in the rate model.
const E: f64 = 1.3;
/// Complexity added to every frame's (mode info and the like).
const C0: f64 = 0.1;
/// The fraction of the reference's quantiser step below which an inter
/// frame of a static scene starts coding the reference's error.
const REFINE: f64 = 0.6;
/// The most of the horizon's budget a key frame is planned to take.
const KEY_SHARE: f64 = 0.1;
/// The frames over which a deviation from the budget is repaid.
const REPAY: f64 = 12.0;
/// The most an inter frame's quantiser index falls below the last inter
/// frame's.
pub(crate) const MAX_FALL: u32 = 8;

/// A frame's class: what the model and the quantiser offset depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    Key,
    Inter,
    /// A frame of an alt-ref group, of this layer of the pyramid.
    Layer(u8),
}

/// The layers of a group's pyramid with models of their own (deeper ones
/// share the last).
const LAYERS: usize = 6;
/// The coarsest quantiser a group's plan reaches: past 255, each step of
/// 32 is an octave more for the frames of the layers finer than the top
/// one, all of them at 255 at the end (a rate below what the top layer
/// costs at 255 could not be reached otherwise).
pub(crate) const MAX_GROUP_Q: u32 = 255 + 64;
/// The first guess of each layer's model against the key frame's, its
/// complexity measured by the lookahead (`lookahead`): the median over the
/// test clips' first groups at quantisers 24 to 240 (a frame predicted
/// across the whole group costs more per unit of its residual than the
/// key frame does per unit of its own; the top layers' residuals are
/// mostly left uncoded).
const LAYER_PRIOR: [f64; LAYERS] = [1.2, 1.2, 1.1, 0.9, 0.6, 0.6];
/// The first guess of the key frame's model.
const KEY_PRIOR: f64 = 25.0;

impl Class {
    /// The class's model.
    fn model(self) -> usize {
        match self {
            Class::Key => 0,
            Class::Inter => 1,
            Class::Layer(l) => 2 + (l as usize).min(LAYERS - 1),
        }
    }
}

/// A frame's complexity (see the module documentation).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Complexity {
    pub(crate) intra: f64,
    pub(crate) inter: f64,
}

/// The frame being coded, as planned.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Planned {
    pub(crate) class: Class,
    pub(crate) qidx: u32,
    /// The plan's quantiser for plain inter frames.
    pub(crate) base: u32,
    pub(crate) c: Complexity,
    pub(crate) predicted: f64,
    /// The quantiser of the reference whose error the frame refines when
    /// it is coded finer (see `refinement`).
    pub(crate) ref_q: Option<u32>,
}

/// What a frame of a group is predicted from, for its refinement term:
/// the finest of the frames already coded it refers to (the group's anchor
/// among them; the frames of the group it refers to not yet coded are
/// coded finer than it, at their layers' offsets), and whether it refers
/// to a key frame planned with the group, not yet coded.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Refs {
    pub(crate) key: bool,
    pub(crate) coded: Option<u32>,
}

impl Refs {
    /// The quantiser refined, the key frame's `key_q`.
    fn q(self, key_q: Option<u32>) -> Option<u32> {
        [key_q.filter(|_| self.key), self.coded]
            .into_iter()
            .flatten()
            .min()
    }
}

pub(crate) struct RateControl {
    /// Frames are coded in alt-ref groups (a key frame's plan may then
    /// reach past 255, as a group's does).
    groups: bool,
    /// Bits per frame on average.
    target: f64,
    bit_depth: u32,
    horizon: usize,
    pixels: f64,
    /// Bits spent and frames coded so far.
    spent: f64,
    frames: u64,
    /// Model constants: key, inter, the group layers; whether fitted.
    k: [f64; 2 + LAYERS],
    fitted: [bool; 2 + LAYERS],
    /// The last inter frame's complexity (the guess for the frames ahead).
    inter_complexity: Option<f64>,
    /// The previous source frame's half-size luma (8-bit scale).
    prev: Option<Vec<u16>>,
    /// The quantiser the reference's samples were (mostly) coded at: a
    /// frame that changes little and is coded coarser than its reference
    /// copies it, so the quality stays the reference's.
    ref_q: Option<u32>,
    /// The plan's quantiser at the last plain inter frame.
    last_inter_q: Option<u32>,
    /// The quantiser the last plan gave plain inter frames.
    pub(crate) base_q: u32,
}

impl RateControl {
    pub(crate) fn new(
        target_bits_per_frame: u64,
        width: u32,
        height: u32,
        bit_depth: u32,
        keyframe_interval: u32,
        start_q: u32,
        groups: bool,
    ) -> Self {
        RateControl {
            groups,
            target: target_bits_per_frame.max(1) as f64,
            bit_depth,
            horizon: (keyframe_interval as usize).clamp(8, 48),
            pixels: width as f64 * height as f64,
            spent: 0.0,
            frames: 0,
            k: std::array::from_fn(|m| match m {
                0 => KEY_PRIOR,
                1 => 0.5 * KEY_PRIOR,
                _ => KEY_PRIOR * LAYER_PRIOR[m - 2],
            }),
            fitted: [false; 2 + LAYERS],
            inter_complexity: None,
            prev: None,
            ref_q: None,
            last_inter_q: None,
            base_q: start_q.clamp(1, 255),
        }
    }

    fn qstep(&self, qidx: u32) -> f64 {
        let bdi = ((self.bit_depth - 8) >> 1) as usize;
        AC_QLOOKUP[bdi][qidx.clamp(0, 255) as usize] as f64 / (1 << (self.bit_depth - 8)) as f64
    }

    /// `pixels / qstep^E` at `qidx`.
    fn scale(&self, qidx: u32) -> f64 {
        self.pixels / self.qstep(qidx).powf(E)
    }

    /// The refinement part of an inter frame's cost at `qidx`: what
    /// coding the reference's error costs once the step is fine enough to
    /// code it (below `REFINE` of the reference's).
    fn refinement(&self, c: &Complexity, qidx: u32, ref_q: Option<u32>) -> f64 {
        let Some(r) = ref_q else {
            return 0.0;
        };
        let rs = (self.qstep(r) * REFINE).powf(-E) * self.pixels;
        let s = self.scale(qidx);
        if s > rs {
            self.k[Class::Key.model()] * (c.intra + C0) * (s - rs)
        } else {
            0.0
        }
    }

    fn bits(&self, class: Class, c: &Complexity, qidx: u32, ref_q: Option<u32>) -> f64 {
        match class {
            Class::Key => self.k[0] * (c.intra + C0) * self.scale(qidx),
            Class::Inter => {
                self.k[1] * (c.inter + C0) * self.scale(qidx) + self.refinement(c, qidx, ref_q)
            }
            // (A frame of a group, coded finer than what it is predicted
            // from, refines it as an inter frame does: the alt-ref frame,
            // the finest of its group, its anchor; a frame planned again
            // finer than the frames coded before it, them.)
            Class::Layer(_) => {
                self.k[class.model()] * (c.inter + C0) * self.scale(qidx)
                    + self.refinement(c, qidx, ref_q)
            }
        }
    }

    /// Measures the source's complexity (luma `y`, `stride`, `w` x `h`, at
    /// the configured bit depth) and, for a frame shown in its turn
    /// (`keep`; not an alt-ref frame coded ahead), keeps its half-size luma
    /// for the next frame's.
    pub(crate) fn measure(
        &mut self,
        y: &[u16],
        stride: usize,
        w: usize,
        h: usize,
        keep: bool,
    ) -> Complexity {
        let shift = self.bit_depth - 8;
        let (hw, hh) = (w / 2, h / 2);
        let mut half = vec![0u16; hw * hh];
        for i in 0..hh {
            let (r0, r1) = (&y[2 * i * stride..], &y[(2 * i + 1) * stride..]);
            for j in 0..hw {
                let s = r0[2 * j] as u32
                    + r0[2 * j + 1] as u32
                    + r1[2 * j] as u32
                    + r1[2 * j + 1] as u32;
                half[i * hw + j] = ((s + 2) >> (2 + shift)) as u16;
            }
        }
        let prev = self.prev.as_ref().filter(|p| p.len() == half.len());
        let (mut intra_t, mut inter_t) = (0f64, 0f64);
        let mut n = 0usize;
        for by in (0..hh.saturating_sub(7)).step_by(8) {
            for bx in (0..hw.saturating_sub(7)).step_by(8) {
                let mut sum = 0u32;
                for i in 0..8 {
                    for j in 0..8 {
                        sum += half[(by + i) * hw + bx + j] as u32;
                    }
                }
                let mean = ((sum + 32) >> 6) as i32;
                let mut intra = 0u32;
                let mut inter = 0u32;
                for i in 0..8 {
                    for j in 0..8 {
                        let o = (by + i) * hw + bx + j;
                        let v = half[o] as i32;
                        intra += (v - mean).unsigned_abs();
                        if let Some(p) = prev {
                            inter += (v - p[o] as i32).unsigned_abs();
                        }
                    }
                }
                intra_t += intra as f64;
                inter_t += if prev.is_some() {
                    intra.min(inter)
                } else {
                    intra
                } as f64;
                n += 64;
            }
        }
        if keep {
            self.prev = Some(half);
        }
        let n = n.max(1) as f64;
        Complexity {
            intra: intra_t / n,
            inter: inter_t / n,
        }
    }

    /// The quantiser of the next frame: its `class`, complexity `c` from
    /// [`Self::measure`], `frame_q(q)` its quantiser when the plan's is `q`
    /// (finer for key frames and the references of more frames, coarser
    /// for the frames no other refers to).
    pub(crate) fn plan(
        &self,
        class: Class,
        c: Complexity,
        frame_q: impl Fn(u32) -> u32,
    ) -> Planned {
        let h = self.horizon as f64;
        let owed = self.frames as f64 * self.target - self.spent;
        // The horizon's budget: each frame's share, plus what earlier
        // frames left or minus what they took, repaid over `REPAY` frames
        // (at most half a share a frame either way: a scene that cannot use
        // its bits, a static one, must not bank them for an overspend the
        // frames after cannot repay).
        let adj = (owed / REPAY).clamp(-0.5 * self.target, 0.5 * self.target);
        let budget = (self.target + adj) * h;
        let future = Complexity {
            intra: c.intra,
            inter: self.inter_complexity.unwrap_or(match class {
                Class::Key => c.intra * 0.3,
                Class::Inter | Class::Layer(_) => c.inter,
            }),
        };
        let cost = |q: u32| -> f64 {
            let qf = frame_q(q);
            // The frames after this one refine it when they are finer.
            self.bits(class, &c, qf, self.ref_q)
                + (h - 1.0) * self.bits(Class::Inter, &future, q, Some(qf))
        };
        // The finest quantiser within the budget; inter frames fall
        // slowly
        let floor = match (class, self.last_inter_q) {
            // (faster while earlier frames have left bits unspent).
            (Class::Inter, Some(l)) => {
                let fall =
                    MAX_FALL + (owed.max(0.0) / self.target).min(3.0 * MAX_FALL as f64) as u32;
                l.saturating_sub(fall).max(1)
            }
            // The first inter frame after a key frame does not refine it.
            (Class::Inter, None) => self.ref_q.unwrap_or(1),
            _ => 1,
        };
        // No frame takes more than a tenth of the horizon's budget (a key
        // frame also half of what earlier frames left unspent): spending
        // stays smooth enough that any stretch of frames comes out near
        // its rate.
        let cap = if class == Class::Key {
            // (and half of what earlier frames left unspent).
            budget * KEY_SHARE + 0.5 * owed.max(0.0)
        } else {
            budget * 0.1
        };
        let max = if self.groups && class == Class::Key {
            MAX_GROUP_Q
        } else {
            255
        };
        let q = (floor..=max)
            .find(|&q| cost(q) <= budget && self.bits(class, &c, frame_q(q), self.ref_q) <= cap)
            .unwrap_or(max);
        let qf = frame_q(q);
        Planned {
            class,
            qidx: qf,
            base: q,
            c,
            predicted: self.bits(class, &c, qf, self.ref_q),
            ref_q: self.ref_q,
        }
    }

    /// The plan's quantiser for the frames of a group still to code:
    /// `group` their layers, complexities and references, `key` the
    /// complexity of the key frame planned with them when it is not yet
    /// coded, `budget` what they are to spend, `frame_q(q, layer)` a frame's
    /// quantiser (`None`: the key frame's) when the plan's is `q`; the plan
    /// falls at most `falls` times `MAX_FALL` (and a little more while
    /// earlier frames have left bits unspent) below the last group's first
    /// plan, and stays at `keep` while that is predicted to spend within
    /// a tenth of the budget.
    pub(crate) fn plan_group(
        &self,
        group: &[(u8, Complexity, Refs)],
        key: Option<Complexity>,
        falls: Option<u32>,
        keep: Option<u32>,
        budget: f64,
        frame_q: impl Fn(u32, Option<u8>) -> u32,
    ) -> u32 {
        let owed = self.owed();
        let cost = |q: u32| -> f64 {
            // (A key frame planned with the group is its anchor.)
            let key_q = key.map(|_| frame_q(q, None));
            let k = key.map_or(0.0, |c| self.bits(Class::Key, &c, frame_q(q, None), None));
            k + group
                .iter()
                .map(|&(l, c, r)| self.bits(Class::Layer(l), &c, frame_q(q, Some(l)), r.q(key_q)))
                .sum::<f64>()
        };
        let floor = self.last_inter_q.zip(falls).map_or(1, |(l, n)| {
            let fall = MAX_FALL + (owed.max(0.0) / self.target).min(3.0 * MAX_FALL as f64) as u32;
            l.saturating_sub(n * fall).max(1)
        });
        if let Some(q) = keep
            && (0.9 * budget..=1.1 * budget).contains(&cost(q))
        {
            return q;
        }
        // (Beyond 255 the finer layers coarsen towards the top one, which
        // stays at 255: see `frame_q`.)
        (floor..=MAX_GROUP_Q)
            .find(|&q| cost(q) <= budget)
            .unwrap_or(MAX_GROUP_Q)
    }

    /// The budget of a group of `n` frames: their shares, plus what earlier
    /// frames left or minus what they took, repaid over the group (at most
    /// half a share a frame left, three quarters taken: see `plan`).
    pub(crate) fn group_budget(&self, n: usize) -> f64 {
        let n = n as f64;
        let adj = (self.owed() / n).clamp(-0.75 * self.target, 0.5 * self.target);
        (self.target + adj) * n
    }

    /// A group's first plan, `q`: where the next group's plan falls from.
    pub(crate) fn start_group(&mut self, q: u32) {
        self.last_inter_q = Some(q);
    }

    /// The quantiser of the last key or alt-ref frame (a group's anchor).
    pub(crate) fn anchor_q(&self) -> Option<u32> {
        self.ref_q
    }

    /// A frame of a group to code again, coarser: it cost twice its
    /// prediction, more than half of what was left of the group's budget,
    /// `left`, and more than four frames' shares (a frame that refines the
    /// noise of the frames before it, which the model did not see; a frame
    /// planned again into refining a still scene). The quantiser at which
    /// it costs its prediction, or two fifths of what was left, by the
    /// model's slope.
    pub(crate) fn redo_layer(&self, p: &Planned, bits: u64, left: f64) -> Option<Planned> {
        let bits = bits as f64;
        if bits <= 2.0 * p.predicted || bits <= (0.5 * left).max(4.0 * self.target) {
            return None;
        }
        let goal = p.predicted.max(0.4 * left);
        let s0 = self.scale(p.qidx);
        let q = (p.qidx..=255)
            .find(|&q| bits * self.scale(q) / s0 <= goal)
            .unwrap_or(255);
        (q > p.qidx).then_some(Planned {
            qidx: q,
            predicted: goal,
            ..*p
        })
    }

    /// Bits spent so far.
    pub(crate) fn spent(&self) -> f64 {
        self.spent
    }

    /// Whether the group should be planned again (and its first frame
    /// coded again): the first frame coded with a layer's model, far from
    /// its prediction. The models no frame has fitted yet are scaled by
    /// what it cost against its prediction.
    pub(crate) fn rescale_layers(&mut self, p: &Planned, bits: u64) -> bool {
        let refined = self.refinement(&p.c, p.qidx, p.ref_q);
        let part = (bits as f64 - refined).max(0.25 * bits as f64);
        let r = part / (p.predicted - refined).max(1.0);
        if (0.75..=1.33).contains(&r) || self.fitted[p.class.model()] {
            return false;
        }
        for m in Class::Layer(0).model()..2 + LAYERS {
            if !self.fitted[m] {
                self.k[m] *= r;
            }
        }
        true
    }

    /// A frame of a group predicted from `refs` (or, without, the key
    /// frame planned with it) planned at the group's quantiser `base`.
    pub(crate) fn planned(
        &self,
        class: Class,
        c: Complexity,
        refs: Option<Refs>,
        base: u32,
        qidx: u32,
    ) -> Planned {
        let ref_q = refs.and_then(|r| r.q(None));
        Planned {
            class,
            qidx,
            base,
            c,
            predicted: self.bits(class, &c, qidx, ref_q),
            ref_q,
        }
    }

    /// Whether a key frame planned with its group should be planned again
    /// with it (and coded again): far from its prediction, the model
    /// refitted to it.
    pub(crate) fn refit_key(&mut self, p: &Planned, bits: u64) -> bool {
        let r = bits as f64 / p.predicted.max(1.0);
        if (0.75..=1.33).contains(&r) {
            return false;
        }
        self.refit(p, bits, true);
        true
    }

    /// A new plan when the frame should be coded again (once): a key frame,
    /// or the first inter frame, far from its predicted cost (the model
    /// refitted to it, the plan made again), or an inter frame that cost four times its prediction
    /// and more than eight shares and what earlier frames left unspent
    /// (coarser by the cost's slope: a static scene's frame that refined
    /// its reference far beyond the plan).
    pub(crate) fn redo(
        &mut self,
        p: &Planned,
        bits: u64,
        frame_q: impl Fn(u32) -> u32,
    ) -> Option<Planned> {
        let bits = bits as f64;
        let r = bits / p.predicted.max(1.0);
        match p.class {
            // (So is the first inter frame's, whose model was a guess.)
            Class::Key | Class::Inter
                if !(0.75..=1.33).contains(&r)
                    && (p.class == Class::Key || !self.fitted[Class::Inter.model()]) =>
            {
                self.refit(p, bits as u64, true);
                Some(self.plan(p.class, p.c, frame_q))
            }
            Class::Inter
                if r > 4.0 && bits > (8.0 * self.target).max(self.owed() + self.target) =>
            {
                // (A scene that has left bits unspent may use them.)
                let goal = p.predicted.max(4.0 * self.target).max(0.75 * self.owed());
                let s0 = self.scale(p.qidx);
                let q = (p.qidx..=255)
                    .find(|&q| bits * self.scale(q) / s0 <= goal)
                    .unwrap_or(255);
                Some(Planned {
                    qidx: q,
                    base: p.base.max(q),
                    predicted: goal,
                    ..*p
                })
            }
            _ => None,
        }
    }

    /// Bits the frames so far left unspent (negative: overspent).
    fn owed(&self) -> f64 {
        self.frames as f64 * self.target - self.spent
    }

    fn refit(&mut self, p: &Planned, bits: u64, replace: bool) {
        let ci = p.class.model();
        let (part, cx) = match p.class {
            Class::Key => (bits as f64, p.c.intra),
            Class::Layer(_) | Class::Inter => (
                (bits as f64 - self.refinement(&p.c, p.qidx, p.ref_q)).max(0.25 * bits as f64),
                p.c.inter,
            ),
        };
        let obs = part.max(1.0) / ((cx + C0) * self.scale(p.qidx));
        self.k[ci] = if let Class::Layer(_) = p.class {
            // (Its guess replaced by at most four times or a quarter of it,
            // then halfway in proportion and at most twice or half: a
            // group's frames are many and small, and one that codes far
            // from the rest, a tenth or ten times its prediction, is not
            // what the frames after it will cost.)
            let r = (obs / self.k[ci]).clamp(0.25, 4.0);
            self.k[ci] * if self.fitted[ci] { r.sqrt() } else { r }
        } else if replace || !self.fitted[ci] {
            obs
        } else {
            0.5 * self.k[ci] + 0.5 * obs
        };
        self.fitted[ci] = true;
        if p.class == Class::Key {
            // Until an inter frame says otherwise, inter frames cost half
            // as much per unit of complexity, and a group's frames as
            // `LAYER_PRIOR` says.
            for m in 1..2 + LAYERS {
                if !self.fitted[m] {
                    self.k[m] = self.k[ci] * if m == 1 { 0.5 } else { LAYER_PRIOR[m - 2] };
                }
            }
        }
        if let Class::Layer(_) = p.class {
            // The plain inter model, which plans the frames after a key
            // frame, follows the group's frames on average.
            let m = Class::Inter.model();
            self.k[m] = if self.fitted[m] {
                0.9 * self.k[m] + 0.1 * obs
            } else {
                obs
            };
            self.fitted[m] = true;
        }
    }

    /// Accounts for a temporal unit that shows a frame coded earlier.
    pub(crate) fn account_shown(&mut self, bits: u64) {
        self.spent += bits as f64;
        self.frames += 1;
    }

    /// Accounts for a coded frame; `plain` an inter frame at the plan's
    /// quantiser (not a golden or alt-ref frame), `shown` whether it is
    /// shown now (an alt-ref frame is shown later).
    pub(crate) fn update(&mut self, p: &Planned, bits: u64, plain: bool, shown: bool) {
        if let Class::Layer(_) = p.class
            && !self.fitted[p.class.model()]
        {
            // The first frame of its layer: the layers no frame has fitted
            // yet are as far from their guesses as it was from its.
            let refined = self.refinement(&p.c, p.qidx, p.ref_q);
            let part = (bits as f64 - refined).max(0.25 * bits as f64);
            let r = (part / (p.predicted - refined).max(1.0)).clamp(0.25, 4.0);
            for m in Class::Layer(0).model()..2 + LAYERS {
                if !self.fitted[m] && m != p.class.model() {
                    self.k[m] *= r;
                }
            }
        }
        self.refit(p, bits, false);
        self.spent += bits as f64;
        self.frames += shown as u64;
        if p.class != Class::Key && shown {
            self.inter_complexity = Some(p.c.inter);
        }
        if plain && p.class == Class::Inter {
            self.last_inter_q = Some(p.base);
        } else if p.class == Class::Key && !self.groups {
            self.last_inter_q = None;
        }
        if let Class::Layer(l) = p.class {
            self.base_q = p.base.min(255);
            if l == 0 {
                // The alt-ref frame is the next group's anchor.
                self.ref_q = Some(p.qidx);
            }
            return;
        }
        let refined = self.refinement(&p.c, p.qidx, self.ref_q);
        self.ref_q = Some(match self.ref_q {
            // Refined (if its cost says it was).
            Some(r) if p.qidx < r => {
                if refined > 0.0 && (bits as f64) < 0.2 * refined {
                    r
                } else {
                    p.qidx
                }
            }
            // Coarser: the new content's share of the frame takes it.
            Some(r) if p.class == Class::Inter => {
                let m = (2.0 * p.c.inter / p.c.intra.max(1e-3)).min(1.0);
                (r as f64 + m * (p.qidx - r) as f64).round() as u32
            }
            _ => p.qidx,
        });
        self.base_q = p.base.min(255);
    }
}
