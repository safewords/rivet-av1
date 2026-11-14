//! An AV1 decoder and encoder.
//!
//! Rust, no C, no system libraries, no build script. Written from the AV1
//! Bitstream & Decoding Process Specification (AOMedia) — the syntax of
//! its section 5, the decoding process of section 7, the parsing process of
//! section 8 and the tables of section 9 — and not translated from any
//! other implementation.
//!
//! # Decoding
//!
//! ```no_run
//! let mut dec = av1::Decoder::new();
//! # let temporal_units: Vec<Vec<u8>> = Vec::new();
//! for tu in &temporal_units {
//!     if let Some(frame) = dec.decode(tu)? {
//!         // frame.plane(0), frame.plane(1), frame.plane(2): Y, U, V
//!     }
//! }
//! # Ok::<(), av1::Error>(())
//! ```
//!
//! On several threads, with several frames decoded at once (their output
//! a few units behind their input; `docs/threading.md`):
//!
//! ```no_run
//! let mut dec = av1::Decoder::new();
//! dec.set_threads(8);
//! dec.set_frames_in_flight(8);
//! # let temporal_units: Vec<Vec<u8>> = Vec::new();
//! for tu in &temporal_units {
//!     dec.queue(tu);
//!     while let Some(frames) = dec.poll() {
//!         for frame in frames? { /* in order */ }
//!     }
//! }
//! while let Some(frames) = dec.wait() {
//!     for frame in frames? { /* the last ones */ }
//! }
//! # Ok::<(), av1::Error>(())
//! ```
//!
//! # Layout
//!
//! - `obu` — OBUs and the sequence header (5.3, 5.5).
//! - `header` — the frame header (5.9).
//! - `symbol`, `cdf` — the symbol decoder and the adaptive CDFs (8.2, 8.3).
//! - [`decoder`] — tiles, mode info, motion vector prediction, prediction,
//!   residual, the loop filter, CDEF, super-resolution, loop restoration,
//!   film grain, reference management; and the rows of a frame, and the
//!   frames of a stream, as tasks for several threads.
//! - `dsp` — inverse transforms.
//! - [`ivf`] — the IVF container.
//! - [`parallel`] — where the decoder's threads come from.
//! - [`y4m`] — YUV4MPEG2 files, raw video in and out.

#![warn(missing_docs)]
// The decoding processes are written as the specification writes them:
// indexed loops over arrays, which keeps them checkable line by line.
#![allow(clippy::needless_range_loop)]

pub(crate) mod bits;
pub(crate) mod cdf;
pub(crate) mod consts;
pub mod decoder;
pub(crate) mod dsp;
pub mod encoder;
pub mod frame;
pub(crate) mod header;
pub mod ivf;
pub(crate) mod obu;
pub mod parallel;
pub(crate) mod symbol;
#[rustfmt::skip]
pub(crate) mod tables;
pub mod y4m;

pub use decoder::{Decoder, annexb_temporal_units};
pub use encoder::{Config, Encoder, Tools};
pub use frame::{
    ChromaFormat, ColorInfo, ContentLightLevel, Frame, HdrMetadata, MasteringDisplay, Plane,
};

/// Errors the decoder and encoder can report.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The bitstream is malformed: a syntax element out of range, data cut
    /// short, a reference to a frame that was never decoded.
    #[error("bitstream error: {0}")]
    Bitstream(String),
    /// The stream is valid but uses a feature this crate does not
    /// implement (yet). The message names it.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The caller's input is unusable: a frame of the wrong size or format,
    /// an invalid configuration, a file that is not IVF.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// `Result` with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    #[cold]
    #[inline(never)]
    pub(crate) fn bitstream(msg: impl Into<String>) -> Self {
        Error::Bitstream(msg.into())
    }
    #[cold]
    #[inline(never)]
    pub(crate) fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }
    #[cold]
    #[inline(never)]
    #[allow(dead_code)]
    pub(crate) fn invalid(msg: impl Into<String>) -> Self {
        Error::InvalidInput(msg.into())
    }
}
