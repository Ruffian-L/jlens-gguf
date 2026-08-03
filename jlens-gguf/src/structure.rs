//! Does the residual geometry have structure? No labels, no human categories.
//!
//! Every gate so far scored the geometry against a category *I* chose — subject, then
//! nearly stance. AUC ≈ 0.50 says "the geometry does not encode my ontology". It says
//! nothing about whether the geometry is organised at all. This module asks the weaker,
//! prior question, and asks it without imposing anything.
//!
//! Four measurements, in increasing order of how much they'd justify building on:
//!
//! 1. **Effective dimensionality** vs a shuffled null. Concentrated spectrum = structure.
//! 2. **Silhouette** vs the same null. Do clusters separate better than chance?
//! 3. **Cross-half generalisation.** Cluster one half, assign the other. Structure that
//!    does not survive an unseen prompt set is memorised noise.
//! 4. **Continuation agreement.** Do points in the same cluster produce more similar
//!    continuations? This is the only one that matters for memory, and its ground truth is
//!    *the model's own behaviour* — not a label anyone assigned.
//!
//! The null throughout is a per-dimension shuffle: each dimension's values are permuted
//! independently across samples. That destroys every correlation between dimensions while
//! preserving each dimension's marginal distribution exactly, so any excess structure in
//! the real data cannot be an artefact of scale, outlier dims, or the baseline.

/// Deterministic k-means++ seeding followed by Lloyd's algorithm.
///
/// Deterministic by construction: the seed drives a SplitMix64 stream, so the same data and
/// seed give the same clustering. A gate that cannot be re-run identically is not a gate.
pub fn kmeans(data: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> Vec<usize> {
    let n = data.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(n);
    let mut state = seed | 1;

    // k-means++: first centre uniform, each subsequent one weighted by squared distance.
    let mut centres: Vec<Vec<f32>> = Vec::with_capacity(k);
    centres.push(data[(next_u64(&mut state) as usize) % n].clone());
    while centres.len() < k {
        let d2: Vec<f32> = data
            .iter()
            .map(|p| {
                centres
                    .iter()
                    .map(|c| sq_dist(p, c))
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        let total: f64 = d2.iter().map(|&v| v as f64).sum();
        if total <= 0.0 {
            centres.push(data[(next_u64(&mut state) as usize) % n].clone());
            continue;
        }
        let mut target = (next_unit(&mut state) as f64) * total;
        let mut chosen = n - 1;
        for (i, &v) in d2.iter().enumerate() {
            target -= v as f64;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centres.push(data[chosen].clone());
    }

    let dim = data[0].len();
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        let mut changed = false;
        for (i, p) in data.iter().enumerate() {
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for (c, centre) in centres.iter().enumerate() {
                let d = sq_dist(p, centre);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            if assign[i] != best {
                changed = true;
                assign[i] = best;
            }
        }
        let mut sums = vec![vec![0f64; dim]; centres.len()];
        let mut counts = vec![0usize; centres.len()];
        for (i, p) in data.iter().enumerate() {
            counts[assign[i]] += 1;
            for (s, v) in sums[assign[i]].iter_mut().zip(p) {
                *s += *v as f64;
            }
        }
        for (c, centre) in centres.iter_mut().enumerate() {
            if counts[c] == 0 {
                continue;
            }
            for (slot, s) in centre.iter_mut().zip(&sums[c]) {
                *slot = (s / counts[c] as f64) as f32;
            }
        }
        if !changed {
            break;
        }
    }
    assign
}

/// Mean silhouette over all points. Range [-1, 1]; higher means better-separated clusters.
///
/// Computed on cosine distance, matching how a retrieval index would actually compare these
/// vectors — a silhouette in Euclidean space would grade a geometry nobody queries.
pub fn silhouette(data: &[Vec<f32>], assign: &[usize]) -> f32 {
    let n = data.len();
    if n < 3 {
        return f32::NAN;
    }
    let k = assign.iter().copied().max().unwrap_or(0) + 1;
    if k < 2 {
        return f32::NAN;
    }
    let mut total = 0f64;
    let mut counted = 0usize;
    for i in 0..n {
        let mut sums = vec![0f64; k];
        let mut counts = vec![0usize; k];
        for j in 0..n {
            if i == j {
                continue;
            }
            sums[assign[j]] += cos_dist(&data[i], &data[j]) as f64;
            counts[assign[j]] += 1;
        }
        if counts[assign[i]] == 0 {
            continue;
        }
        let a = sums[assign[i]] / counts[assign[i]] as f64;
        let b = (0..k)
            .filter(|&c| c != assign[i] && counts[c] > 0)
            .map(|c| sums[c] / counts[c] as f64)
            .fold(f64::INFINITY, f64::min);
        if !b.is_finite() {
            continue;
        }
        let denom = a.max(b);
        if denom > 0.0 {
            total += (b - a) / denom;
            counted += 1;
        }
    }
    if counted == 0 {
        f32::NAN
    } else {
        (total / counted as f64) as f32
    }
}

/// Effective dimensionality: `(Σλ)² / Σλ²` over the covariance spectrum.
///
/// Computed from the Gram matrix, which shares the covariance's non-zero eigenvalues, via
/// trace identities — `Σλ = tr(G)` and `Σλ² = ‖G‖_F²`. No eigensolver needed, which matters
/// because candle has none and d_model is 3840 while n is a couple of hundred.
///
/// Isotropic noise gives PR ≈ min(n, d). Structure concentrates variance and drops it.
pub fn participation_ratio(data: &[Vec<f32>]) -> f32 {
    let n = data.len();
    if n < 2 {
        return f32::NAN;
    }
    let dim = data[0].len();
    let mut mean = vec![0f64; dim];
    for p in data {
        for (m, v) in mean.iter_mut().zip(p) {
            *m += *v as f64 / n as f64;
        }
    }
    let centred: Vec<Vec<f64>> = data
        .iter()
        .map(|p| p.iter().zip(&mean).map(|(v, m)| *v as f64 - m).collect())
        .collect();

    let mut trace = 0f64;
    let mut frob = 0f64;
    for i in 0..n {
        for j in 0..n {
            let g: f64 = centred[i].iter().zip(&centred[j]).map(|(a, b)| a * b).sum();
            if i == j {
                trace += g;
            }
            frob += g * g;
        }
    }
    if frob <= 0.0 {
        return f32::NAN;
    }
    ((trace * trace) / frob) as f32
}

/// Per-dimension shuffle: destroys inter-dimension correlation, preserves each dimension's
/// marginal distribution exactly. The null everything is measured against.
pub fn shuffled_null(data: &[Vec<f32>], seed: u64) -> Vec<Vec<f32>> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let dim = data[0].len();
    let mut out = data.to_vec();
    let mut state = seed | 1;
    for d in 0..dim {
        for i in (1..n).rev() {
            let j = (next_u64(&mut state) as usize) % (i + 1);
            let tmp = out[i][d];
            out[i][d] = out[j][d];
            out[j][d] = tmp;
        }
    }
    out
}

/// Assign `points` to the nearest of `centres`, returning the mean cosine distance to the
/// assigned centre. Lower means the structure learned on one half describes the other.
pub fn assign_cost(points: &[Vec<f32>], centres: &[Vec<f32>]) -> f32 {
    if points.is_empty() || centres.is_empty() {
        return f32::NAN;
    }
    let mut total = 0f64;
    for p in points {
        let best = centres
            .iter()
            .map(|c| cos_dist(p, c))
            .fold(f32::INFINITY, f32::min);
        total += best as f64;
    }
    (total / points.len() as f64) as f32
}

/// Cluster centroids for `assign`.
pub fn centroids(data: &[Vec<f32>], assign: &[usize], k: usize) -> Vec<Vec<f32>> {
    let dim = data.first().map(Vec::len).unwrap_or(0);
    let mut sums = vec![vec![0f64; dim]; k];
    let mut counts = vec![0usize; k];
    for (p, &c) in data.iter().zip(assign) {
        counts[c] += 1;
        for (s, v) in sums[c].iter_mut().zip(p) {
            *s += *v as f64;
        }
    }
    sums.into_iter()
        .zip(&counts)
        .filter(|(_, &c)| c > 0)
        .map(|(s, &c)| s.into_iter().map(|v| (v / c as f64) as f32).collect())
        .collect()
}

/// Jaccard over two token sequences — how much two continuations agree.
pub fn token_overlap(a: &[u32], b: &[u32]) -> f32 {
    use std::collections::HashSet;
    let sa: HashSet<u32> = a.iter().copied().collect();
    let sb: HashSet<u32> = b.iter().copied().collect();
    if sa.is_empty() && sb.is_empty() {
        return 0.0;
    }
    sa.intersection(&sb).count() as f32 / sa.union(&sb).count().max(1) as f32
}

/// Normalised mutual information between a clustering and a labelling, `2·I / (H₁+H₂)`.
///
/// 0 means the clustering tells you nothing about the labels; 1 means they are the same
/// partition up to renaming. Used only to *interpret* clusters after the fact — never to
/// find them, so no human category enters the discovery step.
///
/// Normalised because raw MI grows with the number of clusters, which would make a
/// 40-valued labelling look better than an 8-valued one for free.
pub fn nmi(clusters: &[usize], labels: &[usize]) -> f32 {
    use std::collections::HashMap;
    let n = clusters.len();
    if n == 0 || n != labels.len() {
        return f32::NAN;
    }
    let mut joint: HashMap<(usize, usize), f64> = HashMap::new();
    let mut pa: HashMap<usize, f64> = HashMap::new();
    let mut pb: HashMap<usize, f64> = HashMap::new();
    for (&a, &b) in clusters.iter().zip(labels) {
        *joint.entry((a, b)).or_insert(0.0) += 1.0;
        *pa.entry(a).or_insert(0.0) += 1.0;
        *pb.entry(b).or_insert(0.0) += 1.0;
    }
    let n = n as f64;
    let entropy = |m: &HashMap<usize, f64>| -> f64 {
        -m.values()
            .map(|&c| {
                let p = c / n;
                if p > 0.0 {
                    p * p.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>()
    };
    let ha = entropy(&pa);
    let hb = entropy(&pb);
    let mut mi = 0f64;
    for (&(a, b), &c) in &joint {
        let pab = c / n;
        let p_a = pa[&a] / n;
        let p_b = pb[&b] / n;
        if pab > 0.0 {
            mi += pab * (pab / (p_a * p_b)).ln();
        }
    }
    if ha + hb <= 0.0 {
        return 0.0;
    }
    (2.0 * mi / (ha + hb)) as f32
}

fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

pub fn cos_dist(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    1.0 - dot / (na * nb)
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn next_unit(state: &mut u64) -> f32 {
    ((next_u64(state) >> 40) as f32) / ((1u32 << 24) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blobs() -> Vec<Vec<f32>> {
        // Two tight, well-separated groups in 4-D.
        let mut v = Vec::new();
        for i in 0..10 {
            let j = i as f32 * 0.01;
            v.push(vec![10.0 + j, 0.0, 0.0, 0.0]);
        }
        for i in 0..10 {
            let j = i as f32 * 0.01;
            v.push(vec![0.0, 10.0 + j, 0.0, 0.0]);
        }
        v
    }

    #[test]
    fn kmeans_recovers_obvious_blobs_and_is_deterministic() {
        let data = blobs();
        let a = kmeans(&data, 2, 50, 7);
        let b = kmeans(&data, 2, 50, 7);
        assert_eq!(a, b, "same seed must give the same clustering");
        // Every point in the first blob shares a label; likewise the second.
        assert!(a[..10].iter().all(|&c| c == a[0]));
        assert!(a[10..].iter().all(|&c| c == a[10]));
        assert_ne!(a[0], a[10]);
    }

    #[test]
    fn silhouette_is_high_for_real_blobs_and_low_after_shuffling() {
        let data = blobs();
        let real = silhouette(&data, &kmeans(&data, 2, 50, 7));
        let null = shuffled_null(&data, 7);
        let shuffled = silhouette(&null, &kmeans(&null, 2, 50, 7));
        assert!(real > 0.8, "well-separated blobs should score high, got {real}");
        assert!(
            real > shuffled,
            "real {real} must beat shuffled {shuffled}"
        );
    }

    #[test]
    fn participation_ratio_drops_when_structure_is_present() {
        let data = blobs();
        // Two blobs span ~1 strong direction plus jitter -> low PR.
        let real = participation_ratio(&data);
        assert!(real < 2.5, "structured data should concentrate variance, got {real}");
    }

    #[test]
    fn shuffle_preserves_each_dimensions_values() {
        let data = blobs();
        let null = shuffled_null(&data, 3);
        for d in 0..4 {
            let mut a: Vec<f32> = data.iter().map(|p| p[d]).collect();
            let mut b: Vec<f32> = null.iter().map(|p| p[d]).collect();
            a.sort_by(|x, y| x.partial_cmp(y).unwrap());
            b.sort_by(|x, y| x.partial_cmp(y).unwrap());
            assert_eq!(a, b, "dimension {d} marginal must be preserved exactly");
        }
    }

    #[test]
    fn nmi_is_one_for_identical_partitions_and_zero_for_independent_ones() {
        // Same partition up to renaming.
        assert!((nmi(&[0, 0, 1, 1], &[5, 5, 9, 9]) - 1.0).abs() < 1e-6);
        // Clusters that cut straight across the labels carry no information about them.
        assert!(nmi(&[0, 1, 0, 1], &[7, 7, 7, 7]).abs() < 1e-6);
    }

    #[test]
    fn nmi_does_not_reward_a_labelling_merely_for_having_more_values() {
        // A 4-valued labelling that is independent of the clustering must not beat a
        // 2-valued one that matches it. Raw MI would; normalised MI must not.
        let clusters = [0, 0, 1, 1];
        let matching = [0, 0, 1, 1];
        let finer_but_unrelated = [0, 1, 2, 3];
        assert!(nmi(&clusters, &matching) > nmi(&clusters, &finer_but_unrelated));
    }

    #[test]
    fn token_overlap_is_jaccard() {
        assert_eq!(token_overlap(&[1, 2, 3], &[1, 2, 3]), 1.0);
        assert_eq!(token_overlap(&[1, 2], &[3, 4]), 0.0);
        assert!((token_overlap(&[1, 2], &[2, 3]) - 1.0 / 3.0).abs() < 1e-6);
    }
}
