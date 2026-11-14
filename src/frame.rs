//! The picture type: what the decoder hands back and the encoder takes.
//!
//! The shape follows rivet-vp9's `Frame` (and rivet-h26x's `Picture`): one
//! buffer holding the planes one after the other (Y, then U, then V),
//! tightly packed, one byte per sample at 8 bits and little-endian `u16`
//! above — the layout of a raw planar frame.

/// Chroma sampling of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChromaFormat {
    /// Luma only (`mono_chrome`): one plane.
    Mono,
    /// 4:2:0 — chroma halved in both directions.
    Yuv420,
    /// 4:2:2 — chroma halved horizontally.
    Yuv422,
    /// 4:4:4 — no subsampling.
    Yuv444,
}

impl ChromaFormat {
    /// The bitstream's `(subsampling_x, subsampling_y)` (monochrome reports
    /// 4:2:0's, as the specification does).
    pub fn shifts(self) -> (u32, u32) {
        match self {
            ChromaFormat::Mono | ChromaFormat::Yuv420 => (1, 1),
            ChromaFormat::Yuv422 => (1, 0),
            ChromaFormat::Yuv444 => (0, 0),
        }
    }

    /// From the bitstream's `subsampling_x` and `subsampling_y` (4:4:0 is
    /// not an AV1 format and maps to 4:2:0's neighbour 4:2:2's opposite;
    /// AV1 cannot signal it).
    pub fn from_shifts(ss_x: u32, ss_y: u32) -> Self {
        match (ss_x != 0, ss_y != 0) {
            (true, true) => ChromaFormat::Yuv420,
            (true, false) => ChromaFormat::Yuv422,
            _ => ChromaFormat::Yuv444,
        }
    }

    /// Number of planes: 1 for monochrome, else 3.
    pub fn num_planes(self) -> usize {
        if self == ChromaFormat::Mono { 1 } else { 3 }
    }
}

/// Colour description of a sequence (`color_config()`, 6.4.2): the
/// code points of ITU-T H.273 as AV1 carries them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorInfo {
    /// `color_primaries` (1 = BT.709, 2 = unspecified, 9 = BT.2020, ...).
    pub color_primaries: u32,
    /// `transfer_characteristics` (13 = sRGB, 16 = PQ, 18 = HLG, ...).
    pub transfer_characteristics: u32,
    /// `matrix_coefficients` (0 = identity, 1 = BT.709, 9 = BT.2020 NCL, ...).
    pub matrix_coefficients: u32,
    /// `color_range`: true for full swing, false for studio swing.
    pub full_range: bool,
    /// `chroma_sample_position` (0 unknown, 1 vertical, 2 colocated).
    pub chroma_sample_position: u32,
}

impl Default for ColorInfo {
    fn default() -> Self {
        ColorInfo {
            color_primaries: 2,
            transfer_characteristics: 2,
            matrix_coefficients: 2,
            full_range: false,
            chroma_sample_position: 0,
        }
    }
}

/// Content light level (`metadata_hdr_cll()`, 5.8.3): CTA-861.3's MaxCLL
/// and MaxFALL, in candelas per square metre.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ContentLightLevel {
    /// `max_cll`: the maximum content light level.
    pub max_cll: u16,
    /// `max_fall`: the maximum frame-average light level.
    pub max_fall: u16,
}

/// Mastering display colour volume (`metadata_hdr_mdcv()`, 5.8.4), in the
/// bitstream's fixed-point units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MasteringDisplay {
    /// CIE 1931 `[x, y]` of the red, green and blue primaries, 0.16 fixed
    /// point (50 000 / 65 536 is 0.763; a value of `round(c * 65536)`).
    pub primaries: [[u16; 2]; 3],
    /// CIE 1931 `[x, y]` of the white point, 0.16 fixed point.
    pub white_point: [u16; 2],
    /// Maximum luminance, cd/m², 24.8 fixed point (1000 cd/m² is
    /// `1000 << 8`).
    pub luminance_max: u32,
    /// Minimum luminance, cd/m², 18.14 fixed point (0.005 cd/m² is
    /// `round(0.005 * 16384)` = 82).
    pub luminance_min: u32,
}

/// The high-dynamic-range metadata a stream carries in metadata OBUs
/// (5.8): each part `None` when the stream has not sent it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct HdrMetadata {
    /// `METADATA_TYPE_HDR_CLL`.
    pub content_light: Option<ContentLightLevel>,
    /// `METADATA_TYPE_HDR_MDCV`.
    pub mastering_display: Option<MasteringDisplay>,
}

impl HdrMetadata {
    /// Whether neither part is present.
    pub fn is_empty(&self) -> bool {
        self.content_light.is_none() && self.mastering_display.is_none()
    }
}

/// One plane of a [`Frame`]: where it sits in the frame's data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plane {
    /// Byte offset of the plane's first sample in [`Frame::data`].
    pub offset: usize,
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
}

/// A frame of planar YUV: one buffer holding Y, then U, then V (or Y alone
/// for monochrome), each tightly packed (stride == width).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Luma width in samples.
    pub width: u32,
    /// Luma height in samples.
    pub height: u32,
    /// Bits per sample: 8, 10 or 12.
    pub bit_depth: u32,
    /// Chroma sampling.
    pub chroma: ChromaFormat,
    /// Colour description, as signalled.
    pub color: ColorInfo,
    /// HDR metadata: the most recent content light level and mastering
    /// display metadata OBUs the stream carried before this frame.
    pub hdr: HdrMetadata,
    /// The display size the stream asks for (`render_size`); a hint only.
    pub render_width: u32,
    /// See [`Self::render_width`].
    pub render_height: u32,
    /// Whether the frame is not exactly what the stream codes: a decoder
    /// concealing errors ([`crate::Decoder::set_concealment`]) filled in
    /// what of it did not decode, or predicted it from a frame that was
    /// filled in. Always `false` for a frame decoded without error from
    /// references that were.
    pub concealed: bool,
    /// The samples of every plane, packed: one byte each at 8 bits, else
    /// little-endian `u16` with the value in the low bits.
    pub data: Vec<u8>,
    /// Y, then U, then V.
    pub planes: Vec<Plane>,
}

impl Frame {
    /// A frame of the given geometry with every sample zero.
    pub fn new(width: u32, height: u32, bit_depth: u32, chroma: ChromaFormat) -> Self {
        let (sx, sy) = chroma.shifts();
        let bps = if bit_depth > 8 { 2 } else { 1 };
        let mut planes = Vec::with_capacity(3);
        let mut offset = 0usize;
        for i in 0..chroma.num_planes() {
            let (w, h) = if i == 0 {
                (width, height)
            } else {
                ((width + sx) >> sx, (height + sy) >> sy)
            };
            planes.push(Plane {
                offset,
                width: w,
                height: h,
            });
            offset += w as usize * h as usize * bps;
        }
        Frame {
            width,
            height,
            bit_depth,
            chroma,
            color: ColorInfo::default(),
            hdr: HdrMetadata::default(),
            render_width: width,
            render_height: height,
            concealed: false,
            data: vec![0u8; offset],
            planes,
        }
    }

    /// Bytes per sample, the same for every plane: 1 at 8 bits, else 2.
    pub fn bytes_per_sample(&self) -> usize {
        if self.bit_depth > 8 { 2 } else { 1 }
    }

    fn plane_len(&self, i: usize) -> usize {
        let p = &self.planes[i];
        p.width as usize * p.height as usize * self.bytes_per_sample()
    }

    /// The bytes of plane `i` (0 Y, 1 U, 2 V).
    pub fn plane(&self, i: usize) -> &[u8] {
        let p = self.planes[i];
        &self.data[p.offset..p.offset + self.plane_len(i)]
    }

    /// The bytes of plane `i`, mutably.
    pub fn plane_mut(&mut self, i: usize) -> &mut [u8] {
        let p = self.planes[i];
        let len = self.plane_len(i);
        &mut self.data[p.offset..p.offset + len]
    }

    /// The sample at column `x`, row `y` of plane `i`.
    pub fn sample(&self, i: usize, x: u32, y: u32) -> u16 {
        let p = self.planes[i];
        let idx = y as usize * p.width as usize + x as usize;
        if self.bit_depth > 8 {
            let o = p.offset + 2 * idx;
            u16::from_le_bytes([self.data[o], self.data[o + 1]])
        } else {
            self.data[p.offset + idx] as u16
        }
    }

    /// Sets the sample at column `x`, row `y` of plane `i`.
    pub fn set_sample(&mut self, i: usize, x: u32, y: u32, v: u16) {
        let p = self.planes[i];
        let idx = y as usize * p.width as usize + x as usize;
        if self.bit_depth > 8 {
            let o = p.offset + 2 * idx;
            self.data[o..o + 2].copy_from_slice(&v.to_le_bytes());
        } else {
            self.data[p.offset + idx] = v as u8;
        }
    }

    /// The planes concatenated: Y then U then V.
    pub fn packed(&self) -> &[u8] {
        &self.data
    }

    /// The packed planes, taking the buffer.
    pub fn into_packed(self) -> Vec<u8> {
        self.data
    }
}
