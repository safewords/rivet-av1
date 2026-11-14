//! Several frames in flight: temporal units queued, their frames decoded
//! at the same time as far as they do not depend on each other, and the
//! frames handed back in order, later.
//!
//! `docs/threading.md` has the rules. In short: a unit is *flown* by
//! parsing its headers at once, on the calling thread, as if each of its
//! frames were decoded on the spot (the reference slots take the frames
//! they will hold, 7.20, before those are decoded), while the frames
//! themselves become [`Job`](super::pipeline::Job)s that wait for the
//! frames they refer to. What a unit shows is put together when its frames
//! are complete.
//!
//! Whatever does not fit that (a frame that is not a job, an error of any
//! kind) is not handled here: the decoder goes back to how it was before
//! the unit, waits for everything before it, and decodes the unit as the
//! decoder without frames in flight does. So a unit's result is always
//! that decoder's, whatever went on in flight.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::frame::{Frame, HdrMetadata};
use crate::header::{FilmGrainParams, FrameHeader, RefState};
use crate::obu::SequenceHeader;
use crate::{Decoder, Result};

use super::conformance::Rates;
use super::lst::TileListLayout;
use super::recovery::Recovery;
use super::state::RefData;

/// A frame a flown unit shows: what the output process (7.18) needs once
/// the frame is complete (and whether it is shown by a concealing decoder
/// from damaged references); or the output frame of a tile list (`lst.rs`)
/// once its job has made it, and the list's layout.
pub(super) enum Shown {
    Frame {
        seq: Arc<SequenceHeader>,
        data: Arc<RefData>,
        grain: Box<FilmGrainParams>,
        hdr: HdrMetadata,
        concealed: bool,
    },
    List(ListOut, TileListLayout),
}

/// Where a tile list's job leaves its output frame.
pub(super) type ListOut = Arc<std::sync::Mutex<Option<Frame>>>;

/// What a unit being flown has put in flight and will show.
#[derive(Default)]
pub(super) struct Flight {
    pub(super) shown: Vec<Shown>,
    pub(super) jobs: Vec<u64>,
}

/// The decoder between two units (with no frame begun): what flying a unit
/// changes.
struct Before {
    seq: Option<Arc<SequenceHeader>>,
    ref_state: RefState,
    refs: [Option<Arc<RefData>>; 8],
    seen_frame_header: bool,
    hdr: HdrMetadata,
    /// A frame header in force without tile groups (a large scale tile
    /// stream's camera frame header, which tile lists decode with).
    pending: Option<(Arc<SequenceHeader>, FrameHeader)>,
    recovery: Recovery,
    discard: bool,
    header: (Vec<u8>, usize),
    rates: Rates,
}

enum State {
    /// Not looked at yet.
    Queued,
    /// Its headers are parsed and its frames in flight.
    Flown { before: Box<Before>, flight: Flight },
    /// Decoded: its result.
    Done(Result<Vec<Frame>>),
}

/// A queued temporal unit.
pub(super) struct Unit {
    /// Its bytes, kept until its result is taken: after an error before it
    /// (or in it) it is decoded again from them.
    data: Vec<u8>,
    state: State,
    /// Data was lost before it (`Decoder::signal_loss`).
    loss: bool,
    /// Decoded in place: the layout of the last tile list it decoded, for
    /// `Decoder::tile_list_layout` once its result is taken.
    layout: Option<TileListLayout>,
}

/// How many units are decoded in place without trying to fly them, after
/// one that held a frame that is not a job (a frame of several tiles,
/// say: they come in streams, and the attempt costs a snapshot and a
/// second parse of the unit's headers), and after one that found no
/// thread to fly on (an embedder's threads may still be coming up).
pub(super) const AFTER_A_FRAME_THAT_IS_NOT_A_JOB: usize = 15;
pub(super) const WITHOUT_THREADS: usize = 3;

/// The units queued, oldest first.
#[derive(Default)]
pub(super) struct Units {
    queue: VecDeque<Unit>,
    /// How many of them, from the oldest, are flown or done.
    looked_at: usize,
}

impl Decoder {
    /// Lets up to `n` frames be decoded at the same time, for the units
    /// given to [`Decoder::queue`]. 1, the default, decodes one frame at a
    /// time.
    ///
    /// Frames that do not depend on each other are then decoded side by
    /// side on the decoder's threads ([`Decoder::set_threads`]), and a
    /// frame's parsing does not wait for the frames before it to be
    /// reconstructed: what a stream of one tile per frame needs to use
    /// more than a few threads (see `docs/threading.md`). The frames come
    /// out the same, later: `queue` returns before its unit is decoded,
    /// and [`Decoder::poll`] or [`Decoder::wait`] hand the units' results
    /// back in order. Each frame in flight holds its own buffers, so `n`
    /// bounds the memory: about `n` times what decoding one frame takes.
    ///
    /// [`Decoder::decode`] and [`Decoder::decode_all`] are not affected:
    /// they decode their unit before they return.
    pub fn set_frames_in_flight(&mut self, n: usize) {
        self.frames_in_flight = n.max(1);
    }

    /// Bounds the frames in flight by their size as well as their number
    /// ([`Decoder::set_frames_in_flight`]): no more of them than have
    /// `samples` luma samples between them (one frame always may be,
    /// whatever its size). For a caller that does not know the frame size
    /// when it sets the number: eight frames of 1920x1080 are two of
    /// 3840x2160. No bound by default.
    pub fn set_samples_in_flight(&mut self, samples: u64) {
        self.samples_in_flight = samples;
    }

    /// Queues a temporal unit for decoding. Its result (the frames it
    /// shows, or its error) is the next one [`Decoder::poll`] or
    /// [`Decoder::wait`] gives after those of the units queued before it.
    ///
    /// With one frame in flight ([`Decoder::set_frames_in_flight`]), or
    /// one thread, the unit is decoded before this returns. Otherwise this
    /// returns once the unit's headers are parsed and its frames are on
    /// their way, which may mean waiting (and decoding, on the calling
    /// thread) until an earlier frame is complete, when as many frames are
    /// in flight as there may be.
    pub fn queue(&mut self, unit: &[u8]) {
        let _entered = self.pool.enter();
        let loss = std::mem::take(&mut self.loss_pending);
        self.units.queue.push_back(Unit {
            data: unit.to_vec(),
            state: State::Queued,
            loss,
            layout: None,
        });
        self.look_at_queued();
    }

    /// The number of units queued whose result has not been taken.
    pub fn queued(&self) -> usize {
        self.units.queue.len()
    }

    /// The result of the oldest queued unit, if it is decoded: every frame
    /// it shows, in order, as [`Decoder::decode_all`] returns them, or its
    /// error. `None` if it is still being decoded, or nothing is queued.
    ///
    /// An error is that unit's: the units queued after it are decoded as
    /// if it had been decoded on its own (as after an error of
    /// [`Decoder::decode_all`]).
    pub fn poll(&mut self) -> Option<Result<Vec<Frame>>> {
        let _entered = self.pool.enter();
        self.take_oldest(false)
    }

    /// The result of the oldest queued unit, decoded if need be before
    /// this returns (on the calling thread too). `None` if nothing is
    /// queued.
    pub fn wait(&mut self) -> Option<Result<Vec<Frame>>> {
        let _entered = self.pool.enter();
        self.take_oldest(true)
    }

    /// Forgets the stream: the units queued and their frames in flight,
    /// the reference frames, the sequence header. The decoder is as new,
    /// with its settings and its threads.
    pub fn reset(&mut self) {
        self.engine.abandon_from(0);
        self.engine.end_session();
        self.units = Units::default();
        self.seq = None;
        self.ref_state = RefState::default();
        self.refs = Default::default();
        self.frame = None;
        self.pending = None;
        self.seen_frame_header = false;
        self.shown.clear();
        self.hdr = HdrMetadata::default();
        self.grounded = 0;
        self.recovery = Default::default();
        self.discard = false;
        self.loss_pending = false;
        self.unit_error = None;
        self.rates = Default::default();
    }

    /// Whether the next unit is flown: several frames may be in flight on
    /// several threads, and the decoder is not staying on the ground for
    /// a few units (see `AFTER_A_FRAME_THAT_IS_NOT_A_JOB`).
    fn flies(&mut self) -> bool {
        if self.grounded > 0 {
            self.grounded -= 1;
            return false;
        }
        self.frames_in_flight > 1 && self.threads > 1
    }

    fn before(&self) -> Before {
        Before {
            seq: self.seq.clone(),
            ref_state: self.ref_state.clone(),
            refs: self.refs.clone(),
            seen_frame_header: self.seen_frame_header,
            hdr: self.hdr,
            pending: self.pending.clone(),
            recovery: self.recovery,
            discard: self.discard,
            header: (self.header_bytes.clone(), self.header_bits),
            rates: self.rates.clone(),
        }
    }

    fn go_back(&mut self, b: Before) {
        self.seq = b.seq;
        self.ref_state = b.ref_state;
        self.refs = b.refs;
        self.seen_frame_header = b.seen_frame_header;
        self.hdr = b.hdr;
        self.frame = None;
        self.pending = b.pending;
        self.recovery = b.recovery;
        self.discard = b.discard;
        (self.header_bytes, self.header_bits) = b.header;
        self.rates = b.rates;
    }

    /// Flies (or decodes) every unit not looked at yet, in order.
    fn look_at_queued(&mut self) {
        while self.units.looked_at < self.units.queue.len() {
            let i = self.units.looked_at;
            // A unit that starts inside a frame, or a decoder that does not
            // fly: decoded in place, after everything before it. A frame
            // header in force whose tile groups have not begun (a camera
            // frame header, 7.3, which tile lists decode with) is not
            // inside a frame: a unit after it flies with the header alone.
            let camera = matches!(&self.frame,
                Some(f) if f.tile_num == 0 && f.hdr.refresh_frame_flags == 0);
            if (self.frame.is_some() && !camera) || !self.flies() {
                self.decode_in_place(i);
                continue;
            }
            if camera {
                let f = self.frame.take().expect("the frame header in force");
                self.pending = Some((f.seq.clone(), f.hdr.clone()));
                f.pool.give_back(f.cur);
            }
            let before = Box::new(self.before());
            let data = std::mem::take(&mut self.units.queue[i].data);
            self.flying = Some(Flight::default());
            if self.units.queue[i].loss {
                self.lose();
            }
            let flown = self.decode_obus(&data);
            let flight = self.flying.take().expect("the unit's flight");
            self.shown.clear();
            self.units.queue[i].data = data;
            self.units.queue[i].state = State::Flown { before, flight };
            self.units.looked_at = i + 1;
            if flown.is_err() {
                // Something flying does not do, or an error: the unit is
                // decoded again, in place.
                self.grounded = std::mem::take(&mut self.ground_for);
                self.decode_in_place(i);
            }
        }
    }

    /// Whether the frames unit `i` (flown) put in flight are all complete,
    /// waiting for them if `wait`: `Some(false)` as soon as one of them is
    /// known to have failed (the tile did not parse, or a strict decoder
    /// found its padding wrong), `None` if not known yet.
    fn landed(&mut self, i: usize, wait: bool) -> Option<bool> {
        let State::Flown { flight, .. } = &self.units.queue[i].state else {
            return Some(true);
        };
        for &id in &flight.jobs {
            if wait {
                self.engine.wait_for(id);
            }
            match self.engine.settled(id)? {
                Some(true) => {}
                _ => return Some(false),
            }
        }
        Some(true)
    }

    /// Decodes unit `i` as the decoder without frames in flight decodes
    /// it, once everything before it is decoded: its result is then final.
    /// If a unit before it turns out to have failed, that one is seen to
    /// instead (which sends this one back to the queue).
    fn decode_in_place(&mut self, i: usize) {
        for v in 0..i {
            if self.landed(v, true) == Some(false) {
                return self.decode_failed(v);
            }
        }
        // Whatever the unit itself put in flight is given up, and the
        // decoder is as it was before the unit.
        if let State::Flown { before, flight } =
            std::mem::replace(&mut self.units.queue[i].state, State::Queued)
        {
            if let Some(&first) = flight.jobs.first() {
                self.engine.abandon_from(first);
            }
            self.go_back(*before);
        }
        // Decoding in place has the threads to itself.
        self.engine.end_session();
        if self.units.queue[i].loss {
            self.lose();
        }
        let data = std::mem::take(&mut self.units.queue[i].data);
        let layout = self.anchors.layout.take();
        let result = self.decode_unit(&data);
        self.units.queue[i].layout = self.anchors.layout;
        self.anchors.layout = layout;
        self.units.queue[i].data = data;
        self.units.queue[i].state = State::Done(result);
        self.units.looked_at = i + 1;
    }

    /// Unit `v` (flown, every unit before it fine) has a frame that
    /// failed: everything in flight from it on is given up, the unit is
    /// decoded in place for its error and the state after it, and the
    /// units after it go back to the queue.
    fn decode_failed(&mut self, v: usize) {
        for u in self.units.queue.iter_mut().skip(v + 1) {
            u.state = State::Queued;
        }
        self.units.looked_at = v + 1;
        self.decode_in_place(v);
    }

    fn take_oldest(&mut self, wait: bool) -> Option<Result<Vec<Frame>>> {
        self.look_at_queued();
        if self.units.queue.is_empty() {
            return None;
        }
        match self.landed(0, wait) {
            None => return None,
            Some(false) => {
                self.decode_failed(0);
                // The units after it fly again while this one is taken.
                self.look_at_queued();
            }
            Some(true) => {}
        }
        let unit = self.units.queue.pop_front().expect("the oldest unit");
        self.units.looked_at -= 1;
        Some(match unit.state {
            State::Done(result) => {
                if unit.layout.is_some() {
                    self.anchors.layout = unit.layout;
                }
                result
            }
            State::Flown { flight, .. } => {
                for &id in &flight.jobs {
                    // Asked and answered: the outcome is not kept.
                    let _ = self.engine.outcome(id);
                }
                let mut frames = Vec::with_capacity(flight.shown.len());
                for s in flight.shown {
                    frames.push(match s {
                        Shown::Frame {
                            seq,
                            data,
                            grain,
                            hdr,
                            concealed,
                        } => {
                            let mut f = self.output_with(&seq, &data, &grain, hdr);
                            f.concealed = concealed;
                            f
                        }
                        Shown::List(out, layout) => {
                            self.anchors.layout = Some(layout);
                            out.lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .take()
                                .expect("a tile list's output, once its job is done")
                        }
                    });
                }
                Ok(frames)
            }
            State::Queued => unreachable!("a unit that was looked at"),
        })
    }
}
