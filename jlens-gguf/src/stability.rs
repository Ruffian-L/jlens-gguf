//! The stability gate: is a `dim_signature` key re-hittable?
//!
//! Protocol and binding thresholds are in `docs/jlens-gguf/STABILITY_GATE.md`, written
//! before the first run so the result cannot be graded against whatever came out.
//!
//! Two populations. **Positive pairs** are different paraphrases of the same subject — the
//! same thought opened in different words. **Null pairs** are different subjects. The null
//! is the load-bearing half: a similarity of 0.4 means nothing until you know what two
//! unrelated prompts score.
//!
//! Scored with `weighted_jaccard` from hydro's `jacobian` module — the same function the
//! picker indexes with. Measuring anything else would grade a metric nobody uses.

use std::collections::BTreeMap;

use gguf_hooks::jacobian::{weighted_jaccard, DimSignature};

/// One subject's paraphrases, keyed by subject.
pub type Corpus = BTreeMap<String, Vec<String>>;

/// Similarity distributions for one layer.
#[derive(Debug, Clone)]
pub struct LayerScores {
    pub layer: usize,
    pub positive: Vec<f32>,
    pub null: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct Summary {
    pub layer: usize,
    pub n_positive: usize,
    pub n_null: usize,
    pub median_positive: f32,
    pub median_null: f32,
    /// P(positive pair scores above a null pair). Ties count a half.
    pub auc: f32,
}

impl Summary {
    /// Median positive ÷ median null. Criterion 2.
    pub fn ratio(&self) -> f32 {
        if self.median_null <= 0.0 {
            // A null median of zero means unrelated prompts share nothing at all, which is
            // the best possible separation — report it as such rather than as a NaN.
            return f32::INFINITY;
        }
        self.median_positive / self.median_null
    }
}

impl LayerScores {
    pub fn summarize(&self) -> Summary {
        Summary {
            layer: self.layer,
            n_positive: self.positive.len(),
            n_null: self.null.len(),
            median_positive: median(&self.positive),
            median_null: median(&self.null),
            auc: auc(&self.positive, &self.null),
        }
    }
}

/// Build both populations from per-(subject, paraphrase) signatures.
///
/// `signatures[subject]` holds one signature per paraphrase of that subject.
pub fn score_layer(
    layer: usize,
    signatures: &BTreeMap<String, Vec<DimSignature>>,
    max_null: usize,
) -> LayerScores {
    let mut positive = Vec::new();
    let mut null = Vec::new();

    let subjects: Vec<&String> = signatures.keys().collect();

    // Positive: every within-subject pair.
    for subject in &subjects {
        let sigs = &signatures[*subject];
        for i in 0..sigs.len() {
            for j in (i + 1)..sigs.len() {
                positive.push(weighted_jaccard(&sigs[i], &sigs[j]));
            }
        }
    }

    // Null: cross-subject pairs, strided so the sample spreads over all subject pairs
    // rather than exhausting the first few.
    let mut candidates = Vec::new();
    for a in 0..subjects.len() {
        for b in (a + 1)..subjects.len() {
            let sa = &signatures[subjects[a]];
            let sb = &signatures[subjects[b]];
            for (i, x) in sa.iter().enumerate() {
                for (j, y) in sb.iter().enumerate() {
                    candidates.push((a, b, i, j, x, y));
                }
            }
        }
    }
    let stride = (candidates.len() / max_null.max(1)).max(1);
    for chunk in candidates.iter().step_by(stride) {
        null.push(weighted_jaccard(chunk.4, chunk.5));
        if null.len() >= max_null {
            break;
        }
    }

    LayerScores {
        layer,
        positive,
        null,
    }
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// P(positive > null), ties counted as a half. Computed directly rather than by ranking:
/// the populations are small enough that O(n·m) is free and the definition stays legible.
fn auc(positive: &[f32], null: &[f32]) -> f32 {
    if positive.is_empty() || null.is_empty() {
        return f32::NAN;
    }
    let mut wins = 0f64;
    for p in positive {
        for n in null {
            if p > n {
                wins += 1.0;
            } else if (p - n).abs() < f32::EPSILON {
                wins += 0.5;
            }
        }
    }
    (wins / (positive.len() as f64 * null.len() as f64)) as f32
}

/// Verdict against the thresholds in `STABILITY_GATE.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Criteria 1, 2, 3 met, and L36 (the pre-registered layer) is the passing one.
    Pass,
    /// Criteria met at some layer, but not at the pre-registered L36. The approach holds;
    /// the layer choice was over-fit to the 3-prompt preview.
    PartialPass,
    Fail,
}

pub const AUC_BAR: f32 = 0.80;
pub const RATIO_BAR: f32 = 1.5;
pub const PREREGISTERED_LAYER: usize = 36;

pub fn verdict(summaries: &[Summary], deterministic: bool) -> Verdict {
    if !deterministic {
        return Verdict::Fail;
    }
    let passes = |s: &Summary| s.auc >= AUC_BAR && s.ratio() >= RATIO_BAR;
    let any = summaries.iter().any(passes);
    let preregistered = summaries
        .iter()
        .find(|s| s.layer == PREREGISTERED_LAYER)
        .map(passes)
        .unwrap_or(false);

    match (any, preregistered) {
        (_, true) => Verdict::Pass,
        (true, false) => Verdict::PartialPass,
        (false, false) => Verdict::Fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_is_one_when_populations_are_disjoint() {
        assert!((auc(&[0.8, 0.9], &[0.1, 0.2]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn auc_is_half_when_populations_are_identical() {
        assert!((auc(&[0.5, 0.5], &[0.5, 0.5]) - 0.5).abs() < 1e-6, "ties count a half");
    }

    #[test]
    fn auc_below_half_means_unrelated_pairs_score_higher() {
        // The failure the raw-residual key actually exhibited.
        assert!(auc(&[0.1, 0.2], &[0.8, 0.9]) < 0.5);
    }

    #[test]
    fn verdict_distinguishes_preregistered_layer_from_any_layer() {
        let strong = Summary {
            layer: 36,
            n_positive: 10,
            n_null: 10,
            median_positive: 0.6,
            median_null: 0.2,
            auc: 0.95,
        };
        let elsewhere = Summary { layer: 24, ..strong };
        let weak = Summary {
            layer: 36,
            auc: 0.5,
            median_positive: 0.2,
            median_null: 0.2,
            ..strong
        };

        assert_eq!(verdict(&[strong], true), Verdict::Pass);
        assert_eq!(verdict(&[elsewhere, weak], true), Verdict::PartialPass);
        assert_eq!(verdict(&[weak], true), Verdict::Fail);
        assert_eq!(
            verdict(&[strong], false),
            Verdict::Fail,
            "non-determinism fails the gate whatever the AUC"
        );
    }

    #[test]
    fn ratio_handles_a_zero_null_median() {
        let s = Summary {
            layer: 36,
            n_positive: 1,
            n_null: 1,
            median_positive: 0.4,
            median_null: 0.0,
            auc: 1.0,
        };
        assert!(s.ratio().is_infinite(), "perfect separation is not a NaN");
    }
}
