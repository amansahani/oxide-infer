use anyhow::{anyhow, Result};
use half::f16;
use image::{imageops::FilterType, DynamicImage, GenericImageView};

/// Image preprocessor for LFM3-VL-2B:
/// 1. Resizes image to multiples of patch_size (16)
/// 2. Normalizes RGB values using (val / 255.0 - 0.5) / 0.5 = val / 127.5 - 1.0
/// 3. Patchifies into (grid_h * grid_w, 3 * 2 * 16 * 16 = 1536) in merge-block layout
pub struct VisionPreprocessor {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
}

impl Default for VisionPreprocessor {
    fn default() -> Self {
        Self {
            patch_size: 16,
            merge_size: 2,
            temporal_patch_size: 2,
        }
    }
}

pub struct PatchifiedImage {
    /// Flattened patch embeddings ready for patchifier GEMM:
    /// Length: num_patches * 1536 (f16)
    pub patches: Vec<f16>,
    pub num_patches: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl VisionPreprocessor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Preprocess an in-memory DynamicImage into patch vectors matching official PyTorch layout.
    pub fn process_image(&self, img: &DynamicImage) -> Result<PatchifiedImage> {
        let (raw_w, raw_h) = img.dimensions();
        if raw_w == 0 || raw_h == 0 {
            return Err(anyhow!("Invalid image dimensions: {}x{}", raw_w, raw_h));
        }

        // Align dimensions to multiples of (patch_size * merge_size = 32)
        let block_unit = (self.patch_size * self.merge_size) as u32;
        let target_w = ((raw_w + block_unit - 1) / block_unit) * block_unit;
        let target_h = ((raw_h + block_unit - 1) / block_unit) * block_unit;

        let resized = if target_w != raw_w || target_h != raw_h {
            img.resize_exact(target_w, target_h, FilterType::CatmullRom)
        } else {
            img.clone()
        };

        let rgb = resized.to_rgb8();
        let width = target_w as usize;
        let height = target_h as usize;

        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        let num_patches = grid_h * grid_w;

        // Planar normalization: (C, H, W)
        // val = (byte / 255.0 - 0.5) / 0.5 = (byte as f32) / 127.5 - 1.0
        let mut chw = vec![0.0f32; 3 * height * width];
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let idx = y * width + x;
                chw[0 * height * width + idx] = (pixel[0] as f32) / 127.5 - 1.0;
                chw[1 * height * width + idx] = (pixel[1] as f32) / 127.5 - 1.0;
                chw[2 * height * width + idx] = (pixel[2] as f32) / 127.5 - 1.0;
            }
        }

        // Patchify matching PyTorch:
        // shape: (grid_h * grid_w, 3 * 2 * 16 * 16 = 1536)
        // Layout: (gh//merge, gw//merge, merge_h, merge_w, C, temporal=2, patch_h, patch_w)
        let m = self.merge_size;
        let p = self.patch_size;
        let patch_dim = 3 * self.temporal_patch_size * p * p; // 1536
        let mut patches_f16 = Vec::with_capacity(num_patches * patch_dim);

        let gh_blocks = grid_h / m;
        let gw_blocks = grid_w / m;

        for g_bh in 0..gh_blocks {
            for g_bw in 0..gw_blocks {
                for mh in 0..m {
                    for mw in 0..m {
                        // Current patch position in patch coordinates:
                        let cur_ph = g_bh * m + mh;
                        let cur_pw = g_bw * m + mw;

                        // Fill 1536 elements for this patch:
                        // Channel order (C=0, 1, 2)
                        for c in 0..3 {
                            // Temporal repeat (t=0, 1)
                            for _t in 0..self.temporal_patch_size {
                                for py in 0..p {
                                    for px in 0..p {
                                        let img_y = cur_ph * p + py;
                                        let img_x = cur_pw * p + px;
                                        let val = chw[c * height * width + img_y * width + img_x];
                                        patches_f16.push(f16::from_f32(val));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(PatchifiedImage {
            patches: patches_f16,
            num_patches,
            grid_h,
            grid_w,
        })
    }
}
