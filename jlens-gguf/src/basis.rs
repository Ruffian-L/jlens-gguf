//! Probe bases — the directions the forward-mode fit spends its budget on.
//!
//! Forward mode gives one column of `J` per probe, so an exact fit costs `d_model` probes
//! per (prompt, layer). At `d_model = 3840` that is ~960 batched prefills per prompt per
//! layer with central differences. The default instead probes a rank-`r` subspace.
//!
//! **Why an orthonormalised sample of observed residuals, and not PCA.** The reconstruction
//! is `J h ≈ Y (U h)` with `U` orthonormal and `Y[:, i] = J u_i`. That expression depends
//! only on the *span* of `U`, not on how the basis is rotated inside it — principal axes
//! and any other orthonormal basis of the same span give identical transports. So the
//! principal directions buy nothing here, and skipping them avoids needing an eigensolver
//! candle does not have. What matters is that the span covers where residuals actually
//! live, which a sample of observed residuals does by construction.
//!
//! Components of `h` outside the span are **dropped, not approximated** — see
//! `docs/jlens-gguf/DESIGN.md` §4.

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};

/// How probe directions are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisKind {
    /// `e_0 … e_{d-1}` — the exact fit. `J` comes back column by column.
    Identity,
    /// Orthonormalised sample of residuals observed at this layer.
    ResidualSpan,
    /// Orthonormalised Gaussian directions. A control: it spans the residual subspace no
    /// better than chance, so `ResidualSpan` beating it is evidence the subspace is real.
    Random,
}

impl BasisKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BasisKind::Identity => "identity",
            BasisKind::ResidualSpan => "residual-span",
            BasisKind::Random => "random",
        }
    }
}

/// A set of orthonormal probe directions for one layer.
///
/// `Identity` is held implicitly: materialising a 3840×3840 basis to store `I` would cost
/// 59 MB per layer to say nothing.
pub struct Basis {
    kind: BasisKind,
    d_model: usize,
    rank: usize,
    /// Row-major `[rank, d_model]`. Empty for `Identity`.
    rows: Vec<f32>,
}

impl Basis {
    pub fn identity(d_model: usize) -> Self {
        Self {
            kind: BasisKind::Identity,
            d_model,
            rank: d_model,
            rows: Vec::new(),
        }
    }

    pub fn from_rows(kind: BasisKind, d_model: usize, rows: Vec<f32>) -> Result<Self> {
        if d_model == 0 || rows.len() % d_model != 0 {
            bail!(
                "basis rows ({}) are not a whole multiple of d_model ({d_model})",
                rows.len()
            );
        }
        let rank = rows.len() / d_model;
        Ok(Self {
            kind,
            d_model,
            rank,
            rows,
        })
    }

    pub fn kind(&self) -> BasisKind {
        self.kind
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn is_identity(&self) -> bool {
        self.kind == BasisKind::Identity
    }

    /// Rows `start .. start+len` as a `[len, d_model]` tensor. Identity rows are built on
    /// the fly.
    pub fn chunk(&self, start: usize, len: usize, device: &Device) -> Result<Tensor> {
        let len = len.min(self.rank.saturating_sub(start));
        if len == 0 {
            bail!("empty basis chunk at {start} (rank {})", self.rank);
        }
        let data = if self.is_identity() {
            let mut chunk = vec![0f32; len * self.d_model];
            for (row, slot) in (start..start + len).enumerate() {
                chunk[row * self.d_model + slot] = 1.0;
            }
            chunk
        } else {
            self.rows[start * self.d_model..(start + len) * self.d_model].to_vec()
        };
        Ok(Tensor::from_vec(data, (len, self.d_model), device)?)
    }

    /// The full basis as `[rank, d_model]`. Errors for `Identity`, where callers should
    /// take the `J h` shortcut instead of materialising `I`.
    pub fn rows_tensor(&self, device: &Device) -> Result<Tensor> {
        if self.is_identity() {
            bail!("identity basis is implicit; transport with J directly");
        }
        Ok(Tensor::from_vec(
            self.rows.clone(),
            (self.rank, self.d_model),
            device,
        )?)
    }

    pub fn rows(&self) -> &[f32] {
        &self.rows
    }
}

/// Orthonormalise `samples` (row-major `[n, d]`) and keep up to `rank` rows.
///
/// Modified Gram-Schmidt with a second orthogonalisation pass — "twice is enough" is the
/// standard result, and it matters here because residual samples from one layer are highly
/// correlated, which is exactly the case where a single pass loses orthogonality in f32.
///
/// Rows whose norm collapses under orthogonalisation are dropped: they were already inside
/// the span of earlier rows, so keeping them would inflate the reported rank with
/// directions carrying no new information. The returned basis may therefore be smaller
/// than `rank`, and that is a fact about the samples worth surfacing.
pub fn orthonormalize(samples: &[f32], d_model: usize, rank: usize) -> Result<Vec<f32>> {
    if d_model == 0 || samples.len() % d_model != 0 {
        bail!(
            "samples ({}) are not a whole multiple of d_model ({d_model})",
            samples.len()
        );
    }
    let n = samples.len() / d_model;
    let mut basis: Vec<f32> = Vec::with_capacity(rank * d_model);
    let mut kept = 0usize;

    // Anything below this fraction of the row's original norm is numerical dust.
    const DEPENDENCE_TOL: f32 = 1e-4;

    for i in 0..n {
        if kept == rank {
            break;
        }
        let mut v = samples[i * d_model..(i + 1) * d_model].to_vec();
        let norm0 = l2(&v);
        if norm0 <= f32::EPSILON {
            continue;
        }

        for _pass in 0..2 {
            for j in 0..kept {
                let q = &basis[j * d_model..(j + 1) * d_model];
                let dot = dot(&v, q);
                for (vk, qk) in v.iter_mut().zip(q) {
                    *vk -= dot * qk;
                }
            }
        }

        let norm = l2(&v);
        if norm <= DEPENDENCE_TOL * norm0 {
            continue;
        }
        let inv = 1.0 / norm;
        for vk in v.iter_mut() {
            *vk *= inv;
        }
        basis.extend_from_slice(&v);
        kept += 1;
    }

    if kept == 0 {
        bail!("no independent directions in {n} samples — every residual was degenerate");
    }
    Ok(basis)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Gaussian directions, orthonormalised. Draws `2 × rank` candidates so that dropped
/// dependent rows still leave a full-rank basis.
pub fn random_basis(d_model: usize, rank: usize, seed: u64) -> Result<Vec<f32>> {
    let n = (rank * 2).min(d_model.saturating_mul(2)).max(rank);
    let mut state = seed | 1;
    let mut samples = vec![0f32; n * d_model];
    for slot in samples.iter_mut() {
        *slot = gaussian(&mut state);
    }
    orthonormalize(&samples, d_model, rank)
}

/// Box-Muller over a SplitMix64 stream. Self-contained so a basis is reproducible from
/// its seed alone, independent of whatever `rand` version the workspace resolves to.
fn gaussian(state: &mut u64) -> f32 {
    let u1 = next_unit(state).max(f32::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn next_unit(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32) / ((1u32 << 24) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthonormalize_produces_an_orthonormal_basis() {
        let d = 8;
        let basis = random_basis(d, 4, 42).unwrap();
        assert_eq!(basis.len(), 4 * d);
        for i in 0..4 {
            let bi = &basis[i * d..(i + 1) * d];
            assert!((l2(bi) - 1.0).abs() < 1e-4, "row {i} not unit");
            for j in (i + 1)..4 {
                let bj = &basis[j * d..(j + 1) * d];
                assert!(dot(bi, bj).abs() < 1e-4, "rows {i},{j} not orthogonal");
            }
        }
    }

    #[test]
    fn dependent_rows_are_dropped_not_counted() {
        let d = 4;
        // Second row is 3× the first: one independent direction, not two.
        let samples = vec![
            1.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, 0.0, //
        ];
        let basis = orthonormalize(&samples, d, 3).unwrap();
        assert_eq!(basis.len(), 2 * d, "expected rank 2, got {}", basis.len() / d);
    }

    #[test]
    fn identity_chunks_are_one_hot() {
        let basis = Basis::identity(5);
        assert!(basis.is_identity());
        assert_eq!(basis.rank(), 5);
        let chunk = basis.chunk(2, 2, &Device::Cpu).unwrap();
        let rows: Vec<Vec<f32>> = chunk.to_vec2().unwrap();
        assert_eq!(rows[0], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(rows[1], vec![0.0, 0.0, 0.0, 1.0, 0.0]);
    }
}
