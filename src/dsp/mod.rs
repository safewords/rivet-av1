//! Signal processing kernels, with SIMD versions (AVX2 on x86-64, NEON on
//! aarch64) chosen at run time where they pay, each bit-exact with its
//! scalar version (and tested against it). On wasm32 a module built with
//! the `simd128` target feature takes the SIMD128 version of the CDEF
//! filter and the vector lanes of the inverse transforms; the loop filter
//! and the interpolation filters are loops the compiler vectorises there.

pub(crate) mod cdef;
pub(crate) mod enc;
pub(crate) mod itx;
pub(crate) mod lf;
pub(crate) mod mc;

/// Whether the CPU has AVX2 (checked once). `AV1_NO_SIMD` set in the
/// environment turns the SIMD versions off, for comparison.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn avx2() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on =
                std::is_x86_feature_detected!("avx2") && std::env::var_os("AV1_NO_SIMD").is_none();
            STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
    }
}
