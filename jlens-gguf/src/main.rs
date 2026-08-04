//! `jlens-gguf` — Jacobian lens for GGUF models.
//!
//! See `docs/jlens-gguf/` for the plan, the design derivation, and the changelog.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use clap::{Parser, Subcommand};

use gguf_hooks::loader::{self, Model};
use jlens_gguf::basis::{Basis, BasisKind};
use jlens_gguf::fit::{self, FitConfig, PositionGroups};
use jlens_gguf::keys;
use jlens_gguf::lens::{BandId, Lens};
use jlens_gguf::probe::{band_mask, Site};

#[derive(Parser)]
#[command(name = "jlens-gguf", about = "Jacobian lens for GGUF models")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Args)]
struct ModelArgs {
    #[arg(long)]
    model: String,
    #[arg(long)]
    tokenizer: Option<String>,
    /// Run on CPU even when CUDA is available.
    #[arg(long)]
    cpu: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Load a model and print what the lens will see. No fitting.
    Info {
        #[command(flatten)]
        model: ModelArgs,
    },

    /// Fit a lens and write it to disk.
    Fit {
        #[command(flatten)]
        model: ModelArgs,
        /// Text file, one prompt per line. Blank lines are skipped.
        #[arg(long)]
        prompts: PathBuf,
        /// `all`, `last`, `stride:N`, or a comma list like `8,16,24`.
        #[arg(long, default_value = "stride:6")]
        layers: String,
        /// Probe directions per layer. Omit for the exact fit (rank == d_model).
        #[arg(long, default_value_t = 256)]
        rank: usize,
        /// `residual-span` (default), `random`, or `identity` (exact).
        #[arg(long, default_value = "residual-span")]
        basis: String,
        /// Probe directions carried per forward pass.
        #[arg(long, default_value_t = 8)]
        probe_batch: usize,
        /// ε as a fraction of the source residual's RMS.
        #[arg(long, default_value_t = 1e-2)]
        eps_rel: f32,
        #[arg(long, default_value_t = 128)]
        max_seq_len: usize,
        /// `paper` (default), `thirds`, or `labels:<file.json>`.
        #[arg(long, default_value = "paper")]
        position_groups: String,
        #[arg(long)]
        out: PathBuf,
    },

    /// Collect per-layer residual mean/sd over a corpus.
    ///
    /// Required before `readout` keys on anything meaningful: the top-magnitude dimensions
    /// of a raw residual are the model's constant outlier dimensions, identical for every
    /// prompt. See `baseline.rs` for the measurement that showed it.
    Baseline {
        #[command(flatten)]
        model: ModelArgs,
        /// Text file, one prompt per line.
        #[arg(long)]
        prompts: PathBuf,
        #[arg(long, default_value = "stride:6")]
        layers: String,
        #[arg(long, default_value_t = 512)]
        max_seq_len: usize,
        /// Accumulate only the commit position (last token - offset) instead of every
        /// valid position.
        ///
        /// The baseline must be **position-matched** to the readout. Averaging over all
        /// positions and then subtracting that from a last-position residual leaves a large
        /// position-specific common component: measured cosine between two supposedly
        /// centred commit residuals was 0.74 for unrelated prompts.
        #[arg(long)]
        commit_only: bool,
        #[arg(long, default_value_t = 0)]
        commit_offset: usize,
        #[arg(long)]
        out: PathBuf,
    },

    /// Emit logit-lens telemetry — no fit, no differencing, works today.
    ///
    /// Captures mid-layer residuals and unembeds them directly. This is
    /// `jlens.apply(use_jacobian=False)`: the paper's own baseline, weaker than the fitted
    /// transport but exact arithmetic and unaffected by the quantisation obstruction that
    /// blocks fitting. Every record carries all three addresses — see `telemetry.rs`.
    Readout {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        prompt: String,
        /// `all`, `last`, `stride:N`, or a comma list.
        #[arg(long, default_value = "stride:6")]
        layers: String,
        /// Comma list; negatives count from the end.
        #[arg(long, default_value = "-1", allow_hyphen_values = true)]
        positions: String,
        #[arg(long, default_value_t = 8)]
        top_k: usize,
        /// Dimensions kept in the within-model fingerprint.
        #[arg(long, default_value_t = 32)]
        top_dims: usize,
        /// Append JSONL here instead of stdout.
        #[arg(long)]
        emit: Option<PathBuf>,
        /// Save residual slices here and reference them from each record.
        #[arg(long)]
        state_dir: Option<PathBuf>,
        /// Free-form tag carried into every record (turn id, episode, phase).
        #[arg(long)]
        tag: Option<String>,
        #[arg(long, default_value_t = 512)]
        max_seq_len: usize,
        /// Baseline to subtract. Without it the key ranks the model's furniture.
        #[arg(long)]
        baseline: Option<PathBuf>,
        /// Also run the model's own `forward()` and print its top-k.
        ///
        /// Unembedding the last block's output must reproduce the model's real logits
        /// exactly — same norm, same head, same tensor. If it doesn't, the capture or the
        /// unembed is wrong and every record from this run is fiction.
        #[arg(long)]
        verify: bool,
    },

    /// Read out a fitted lens on a prompt.
    Apply {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        lens: PathBuf,
        #[arg(long)]
        prompt: String,
        /// Comma list; negatives count from the end, as in the reference.
        #[arg(long, default_value = "-2", allow_hyphen_values = true)]
        positions: String,
        #[arg(long, default_value = "all")]
        band: String,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },

    /// **Stability gate.** Are `dim_signature` keys re-hittable?
    ///
    /// Protocol and binding thresholds: `docs/jlens-gguf/STABILITY_GATE.md`, written before
    /// the first run. Scores paraphrases of the same subject (positive) against different
    /// subjects (null) and reports AUC.
    Stability {
        #[command(flatten)]
        model: ModelArgs,
        /// JSON: `{"subject": ["paraphrase", ...], ...}`.
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long, default_value = "24,28,32,36,40")]
        layers: String,
        #[arg(long, default_value_t = 32)]
        top_dims: usize,
        #[arg(long, default_value_t = 8)]
        top_k: usize,
        /// Cap on sampled null pairs.
        #[arg(long, default_value_t = 4000)]
        max_null: usize,
        #[arg(long, default_value_t = 512)]
        max_seq_len: usize,
        /// Tokens back from the end to read. 0 = the commit hinge (last prompt token).
        ///
        /// Every prompt ends with the same template suffix, so the hinge residual may be
        /// dominated by "how do I open a reply" rather than by the subject. Sweeping this
        /// backwards walks into the question's own content.
        #[arg(long, default_value_t = 0)]
        commit_offset: usize,
        /// Content depths inside the thought stream to capture at.
        ///
        /// When set, the gate decodes into the model's own thought block instead of
        /// reading the prefill hinge, and `--baseline` / `--commit-offset` are ignored:
        /// the baseline is computed in-run per (layer, depth), so it is position-matched
        /// by construction. N counts *content* tokens after the channel header closes.
        #[arg(long, value_delimiter = ',')]
        decode_ns: Option<Vec<usize>>,
        /// Decode budget per prompt.
        #[arg(long, default_value_t = 96)]
        max_steps: usize,
        /// Write the full per-pair scores here.
        #[arg(long)]
        emit: Option<PathBuf>,
    },

    /// Does the residual geometry have structure? No labels, no categories.
    ///
    /// Every earlier gate scored the geometry against a category chosen by hand. This asks
    /// the prior question — is it organised at all — against a per-dimension shuffled null.
    Structure {
        #[command(flatten)]
        model: ModelArgs,
        /// Text file, one prompt per line (`\n` for embedded newlines).
        #[arg(long)]
        prompts: PathBuf,
        #[arg(long, default_value = "24,36,44")]
        layers: String,
        /// Content depth inside the thought stream to capture at.
        #[arg(long, default_value_t = 0)]
        depth: usize,
        #[arg(long, default_value_t = 8)]
        k: usize,
        /// Continuation tokens compared for the behaviour test.
        #[arg(long, default_value_t = 12)]
        continuation: usize,
        #[arg(long, default_value_t = 512)]
        max_seq_len: usize,
        #[arg(long, default_value_t = 64)]
        max_steps: usize,
        /// TSV aligned with the prompts file: header row of label names, then one row per
        /// prompt. Used only to *interpret* clusters after they are found — never to find
        /// them, so no human category enters the discovery step.
        #[arg(long)]
        tags: Option<PathBuf>,
    },

    /// **Gate 2.** Sweep ε at one layer and report whether `J v` has a plateau.
    ///
    /// Too small and quantisation noise dominates; too large and the linearisation fails.
    /// A lens fitted off the plateau is not measuring what it claims to.
    Sweep {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        layer: usize,
        /// Comma list of relative ε values.
        #[arg(long, default_value = "1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1")]
        eps: String,
        /// Probe directions to average the comparison over.
        #[arg(long, default_value_t = 4)]
        probes: usize,
        #[arg(long, default_value_t = 128)]
        max_seq_len: usize,
    },

    /// **Gate 1.** Fit with source == target and report `‖J − I‖ / ‖I‖`.
    ///
    /// Catches sign errors, off-by-one in the position mask, the 1/(2ε·|band|) scale, and
    /// batch aliasing between probe directions. Note what it does *not* cover: with source
    /// == target the capture sees the injected value directly, so no transformer block is
    /// exercised and the expected result is algebraically exact. Necessary, not sufficient
    /// — `sweep` is the gate that tests propagation.
    Identity {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        prompt: String,
        /// Probe directions. Small is fine — the check is per-column.
        #[arg(long, default_value_t = 16)]
        probes: usize,
        #[arg(long, default_value_t = 1e-2)]
        eps_rel: f32,
        #[arg(long, default_value_t = 128)]
        max_seq_len: usize,
    },
}

fn pick_device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    match Device::new_cuda(0) {
        Ok(d) => Ok(d),
        Err(e) => {
            eprintln!("cuda unavailable ({e}); falling back to CPU");
            Ok(Device::Cpu)
        }
    }
}

fn load(args: &ModelArgs) -> Result<(Model, tokenizers::Tokenizer, Device)> {
    let device = pick_device(args.cpu)?;
    let loaded = loader::load_gguf(&args.model, args.tokenizer.clone(), &device, true)?;
    Ok((loaded.model, loaded.tokenizer, device))
}

/// `all` | `last` | `stride:N` | `a,b,c`
fn parse_layers(spec: &str, n_layers: usize) -> Result<Vec<usize>> {
    let layers: Vec<usize> = match spec {
        "all" => (0..n_layers).collect(),
        "last" => vec![n_layers - 1],
        other => match other.strip_prefix("stride:") {
            Some(n) => {
                let n: usize = n.parse().context("stride must be a positive integer")?;
                if n == 0 {
                    bail!("stride must be at least 1");
                }
                (0..n_layers).step_by(n).collect()
            }
            None => other
                .split(',')
                .map(|s| s.trim().parse::<usize>().context("layer must be an integer"))
                .collect::<Result<Vec<_>>>()?,
        },
    };
    if layers.is_empty() {
        bail!("--layers {spec:?} selected no layers");
    }
    if let Some(&bad) = layers.iter().find(|&&l| l >= n_layers) {
        bail!("layer {bad} out of range for a {n_layers}-layer model");
    }
    Ok(layers)
}

fn parse_basis(spec: &str) -> Result<BasisKind> {
    match spec {
        "residual-span" => Ok(BasisKind::ResidualSpan),
        "random" => Ok(BasisKind::Random),
        "identity" | "exact" => Ok(BasisKind::Identity),
        other => bail!("unknown --basis {other:?}; expected residual-span, random, or identity"),
    }
}

fn read_prompts(path: &PathBuf) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading prompts from {}", path.display()))?;
    // One prompt per line, but chat templates are multi-line, so a literal `\n` in the
    // file becomes a real newline. `\\n` escapes it back to a literal backslash-n.
    let prompts: Vec<String> = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.replace("\\\\n", "\u{0}").replace("\\n", "\n").replace('\u{0}', "\\n"))
        .collect();
    if prompts.is_empty() {
        bail!("{} contains no prompts", path.display());
    }
    Ok(prompts)
}

/// Tokenise, then set up the position bookkeeping the estimator needs.
fn prepare_prompt(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    max_seq_len: usize,
    device: &Device,
) -> Result<(Tensor, usize, Vec<usize>, Tensor)> {
    let encoded = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
    let mut ids = encoded.get_ids().to_vec();
    ids.truncate(max_seq_len);
    let seq_len = ids.len();
    let tokens = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;
    let valid = fit::valid_positions(seq_len, fit::SKIP_FIRST_N_POSITIONS)?;
    let target_idx = Tensor::new(
        valid.iter().map(|&p| p as u32).collect::<Vec<_>>().as_slice(),
        device,
    )?;
    Ok((tokens, seq_len, valid, target_idx))
}

/// RMS of the residual at `site` over `positions` — the scale relative ε multiplies.
fn source_rms(
    model: &mut Model,
    tokens: &Tensor,
    site: Site,
    positions: &[usize],
    device: &Device,
) -> Result<f32> {
    use gguf_hooks::hooks::LayerHook;
    let mut hook = jlens_gguf::probe::CaptureHook::new([site]);
    model.clear_kv_cache();
    model.forward_with_hidden_hooked(tokens, 0, Some(&mut hook as &mut dyn LayerHook))?;
    let h = hook
        .take(site)
        .ok_or_else(|| anyhow::anyhow!("no activation captured at {site:?}"))?;
    let idx = Tensor::new(
        positions.iter().map(|&p| p as u32).collect::<Vec<_>>().as_slice(),
        device,
    )?;
    let selected = h.narrow(0, 0, 1)?.index_select(&idx, 1)?.to_dtype(DType::F32)?;
    Ok(selected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Info { model } => cmd_info(&model),
        Command::Fit {
            model,
            prompts,
            layers,
            rank,
            basis,
            probe_batch,
            eps_rel,
            max_seq_len,
            position_groups,
            out,
        } => cmd_fit(
            &model,
            &prompts,
            &layers,
            rank,
            &basis,
            probe_batch,
            eps_rel,
            max_seq_len,
            &position_groups,
            &out,
        ),
        Command::Baseline {
            model,
            prompts,
            layers,
            max_seq_len,
            commit_only,
            commit_offset,
            out,
        } => cmd_baseline(
            &model, &prompts, &layers, max_seq_len, commit_only, commit_offset, &out,
        ),
        Command::Readout {
            model,
            prompt,
            layers,
            positions,
            top_k,
            top_dims,
            emit,
            state_dir,
            tag,
            max_seq_len,
            baseline,
            verify,
        } => cmd_readout(
            &model,
            &prompt,
            &layers,
            &positions,
            top_k,
            top_dims,
            emit.as_deref(),
            state_dir.as_deref(),
            tag,
            max_seq_len,
            baseline.as_deref(),
            verify,
        ),
        Command::Apply {
            model,
            lens,
            prompt,
            positions,
            band,
            top_k,
        } => cmd_apply(&model, &lens, &prompt, &positions, &band, top_k),
        Command::Stability {
            model,
            corpus,
            baseline,
            layers,
            top_dims,
            top_k,
            max_null,
            max_seq_len,
            commit_offset,
            decode_ns,
            max_steps,
            emit,
        } => match decode_ns {
            Some(ns) => cmd_stability_decode(
                &model, &corpus, &layers, &ns, top_dims, top_k, max_null, max_seq_len,
                max_steps, emit.as_deref(),
            ),
            None => cmd_stability(
                &model, &corpus, &baseline, &layers, top_dims, top_k, max_null, max_seq_len,
                commit_offset, emit.as_deref(),
            ),
        },
        Command::Structure {
            model,
            prompts,
            layers,
            depth,
            k,
            continuation,
            max_seq_len,
            max_steps,
            tags,
        } => cmd_structure(
            &model, &prompts, &layers, depth, k, continuation, max_seq_len, max_steps,
            tags.as_deref(),
        ),
        Command::Sweep {
            model,
            prompt,
            layer,
            eps,
            probes,
            max_seq_len,
        } => cmd_sweep(&model, &prompt, layer, &eps, probes, max_seq_len),
        Command::Identity {
            model,
            prompt,
            probes,
            eps_rel,
            max_seq_len,
        } => cmd_identity(&model, &prompt, probes, eps_rel, max_seq_len),
    }
}

fn cmd_info(args: &ModelArgs) -> Result<()> {
    let (model, _tokenizer, _device) = load(args)?;
    println!("variant      {}", model.variant_name());
    println!("n_layers     {}", model.n_layers());
    println!("d_model      {}", model.token_embeddings().dim(1)?);
    println!("embed scale  {:.4}", model.embedding_input_scale());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_fit(
    args: &ModelArgs,
    prompts_path: &PathBuf,
    layers_spec: &str,
    rank: usize,
    basis_spec: &str,
    probe_batch: usize,
    eps_rel: f32,
    max_seq_len: usize,
    groups_spec: &str,
    out: &PathBuf,
) -> Result<()> {
    let prompts = read_prompts(prompts_path)?;
    let (mut model, tokenizer, device) = load(args)?;
    let n_layers = model.n_layers();
    let d_model = model.token_embeddings().dim(1)?;

    let cfg = FitConfig {
        source_layers: parse_layers(layers_spec, n_layers)?,
        rank: Some(rank.min(d_model)),
        basis_kind: parse_basis(basis_spec)?,
        probe_batch: probe_batch.max(1),
        eps_rel,
        max_seq_len,
        skip_first: fit::SKIP_FIRST_N_POSITIONS,
        groups: PositionGroups::parse(groups_spec)?,
        seed: 0,
    };

    println!(
        "\nfit: {} prompts, layers {:?}, rank {} ({}), groups {}, eps_rel {:.1e}",
        prompts.len(),
        cfg.source_layers,
        cfg.rank.unwrap_or(d_model),
        cfg.basis_kind.as_str(),
        cfg.groups.as_str(),
        cfg.eps_rel,
    );

    println!("\nbuilding probe bases...");
    let bases = fit::build_bases(&mut model, &tokenizer, &prompts, &cfg, &device, true)?;

    println!("\nfitting...");
    let total = prompts.len();
    let lens = fit::fit(
        &mut model,
        &tokenizer,
        &prompts,
        &cfg,
        &bases,
        &device,
        |idx, stats| {
            println!(
                "  prompt {}/{}  seq_len={} n_valid={}  {:.0}s  max||J||/sqrt(d)={:.4}",
                idx + 1,
                total,
                stats.seq_len,
                stats.n_valid,
                stats.seconds,
                stats.norm_over_sqrt_d,
            );
        },
    )?;

    lens.save(out)?;
    println!(
        "\nwrote {} ({} transports over {} prompts)",
        out.display(),
        lens.blocks.len(),
        lens.n_prompts
    );
    Ok(())
}

/// Unsupervised structure test — no labels anywhere.
#[allow(clippy::too_many_arguments)]
fn cmd_structure(
    args: &ModelArgs,
    prompts_path: &PathBuf,
    layers_spec: &str,
    depth: usize,
    k: usize,
    continuation: usize,
    max_seq_len: usize,
    max_steps: usize,
    tags_path: Option<&std::path::Path>,
) -> Result<()> {
    use jlens_gguf::structure as st;

    let prompts = read_prompts(prompts_path)?;
    let (mut model, tokenizer, device) = load(args)?;
    let n_layers = model.n_layers();
    let layers = parse_layers(layers_spec, n_layers)?;
    let eos: Vec<u32> = vec![1, 106, 50];
    // Capture at `depth`, and far enough beyond it to compare continuations.
    let depths: Vec<usize> = (0..=(depth + continuation)).collect();

    println!(
        "\nstructure test: {} prompts, layers {layers:?}, depth {depth}, k={k}",
        prompts.len()
    );
    println!("null = per-dimension shuffle (destroys correlations, keeps marginals)\n");

    let mut residuals: BTreeMap<usize, Vec<Vec<f32>>> = BTreeMap::new();
    let mut continuations: Vec<Vec<u32>> = Vec::new();
    let mut used: Vec<usize> = Vec::new();

    for (i, prompt) in prompts.iter().enumerate() {
        let ids = jlens_gguf::tokens::encode_prompt(&tokenizer, prompt, max_seq_len)?;
        // The thought stream is already open when the prompt ends with the channel
            // header, and also when the model family has no thought channel at all
            // (Gemma 3 answers directly after `<start_of_turn>model`). Only a Gemma 4
            // prompt that stops before the header has a header left to skip.
            let header_prefilled =
                prompt.trim_end().ends_with("<channel|>") || !prompt.contains("channel");

        let captured = jlens_gguf::decode::decode_and_capture(
            &mut model,
            &tokenizer,
            &ids,
            &layers,
            &depths,
            max_steps,
            &eos,
            header_prefilled,
            &device,
        )?;
        if captured.anchor_step.is_none() || captured.content.len() <= depth {
            continue;
        }
        let mut ok = true;
        for &layer in &layers {
            if !captured.residuals.contains_key(&(layer, depth)) {
                ok = false;
            }
        }
        if !ok {
            continue;
        }
        for &layer in &layers {
            let r = &captured.residuals[&(layer, depth)];
            residuals.entry(layer).or_default().push(r.to_vec1::<f32>()?);
        }
        // Continuation = content tokens strictly after the captured moment, so the
        // behaviour test never scores the geometry against the token it just produced.
        continuations.push(captured.content.iter().skip(depth + 1).copied().collect());
        used.push(i);

        if (i + 1) % 20 == 0 {
            println!("  {}/{} prompts", i + 1, prompts.len());
        }
    }

    let n = continuations.len();
    if n < 20 {
        bail!("only {n} usable prompts; need at least 20 for the null to mean anything");
    }
    println!("\n{n} usable prompts\n");

    for &layer in &layers {
        let data = &residuals[&layer];
        let null = st::shuffled_null(data, 11);

        let pr_real = st::participation_ratio(data);
        let pr_null = st::participation_ratio(&null);

        let a_real = st::kmeans(data, k, 60, 7);
        let a_null = st::kmeans(&null, k, 60, 7);
        let sil_real = st::silhouette(data, &a_real);
        let sil_null = st::silhouette(&null, &a_null);

        // Cross-half: cluster the first half, score the second against those centroids,
        // and compare with the same points scored against the *other* half's centroids
        // shuffled. Structure that does not transfer is memorised noise.
        let half = n / 2;
        let (first, second) = data.split_at(half);
        let a_first = st::kmeans(first, k, 60, 7);
        let cent = st::centroids(first, &a_first, k);
        let cost_real = st::assign_cost(second, &cent);
        let null_first = st::shuffled_null(first, 13);
        let cent_null = st::centroids(&null_first, &st::kmeans(&null_first, k, 60, 7), k);
        let cost_null = st::assign_cost(second, &cent_null);

        println!("L{layer}");
        println!(
            "  effective dim      real {pr_real:8.2}   shuffled {pr_null:8.2}   {}",
            verdict(pr_real < pr_null * 0.9)
        );
        println!(
            "  silhouette         real {sil_real:8.4}   shuffled {sil_null:8.4}   {}",
            verdict(sil_real > sil_null + 0.02)
        );
        println!(
            "  cross-half cost    real {cost_real:8.4}   shuffled {cost_null:8.4}   {}",
            verdict(cost_real < cost_null * 0.98)
        );

        // Behaviour test: do same-cluster points continue more alike than different-cluster
        // points? Ground truth is the model's own output, not a label anyone assigned.
        let mut same = Vec::new();
        let mut diff = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let overlap = st::token_overlap(&continuations[i], &continuations[j]);
                if a_real[i] == a_real[j] {
                    same.push(overlap);
                } else {
                    diff.push(overlap);
                }
            }
        }
        let m_same = mean(&same);
        let m_diff = mean(&diff);
        println!(
            "  continuation agree same-cluster {m_same:.4}   other {m_diff:.4}   ratio {:.2}x   {}",
            if m_diff > 0.0 { m_same / m_diff } else { f32::NAN },
            verdict(m_same > m_diff * 1.15)
        );
        println!();
    }

    // Show what the clusters actually contain. The metrics say structure exists; only
    // looking at the members says what it is.
    {
        let show = layers[layers.len() / 2];
        let assign = st::kmeans(&residuals[&show], k, 60, 7);
        println!("cluster exemplars at L{show} (continuation after the captured moment)\n");
        for c in 0..k {
            let members: Vec<usize> = (0..n).filter(|&i| assign[i] == c).collect();
            if members.is_empty() {
                continue;
            }
            println!("  cluster {c}  ({} prompts)", members.len());
            for &i in members.iter().take(3) {
                let text = tokenizer
                    .decode(&continuations[i], false)
                    .unwrap_or_default();
                // Don't trim: Gemma 3 opens with "\n\n" and trimming shows an empty
                // exemplar for every member of a real cluster.
                let text: String = text.chars().take(72).collect();
                println!("    {:?}", text);
            }
        }
        println!();
    }

    if let Some(path) = tags_path {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading tags {}", path.display()))?;
        let mut lines = raw.lines();
        let header: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
        let rows: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
        println!("what are the clusters? (NMI: 0 = unrelated, 1 = same partition)\n");
        println!("{:>6}  {}", "layer", header.join("   "));
        for &layer in &layers {
            let assign = st::kmeans(&residuals[&layer], k, 60, 7);
            let mut cells = Vec::new();
            for (col, _name) in header.iter().enumerate() {
                let mut ids = std::collections::HashMap::new();
                let labels: Vec<usize> = used
                    .iter()
                    .map(|&i| {
                        let v = rows.get(i).and_then(|r| r.get(col)).copied().unwrap_or("");
                        let next = ids.len();
                        *ids.entry(v.to_string()).or_insert(next)
                    })
                    .collect();
                cells.push(format!("{:.3}", st::nmi(&assign, &labels)));
            }
            println!("L{:<5}  {}", layer, cells.join("   "));
        }
        println!();
    }

    println!(
        "PASS on `continuation agree` is the one that matters: it means the geometry\n\
         predicts what the model does next, with no human category anywhere in the loop."
    );
    Ok(())
}

fn verdict(pass: bool) -> &'static str {
    if pass {
        "STRUCTURE"
    } else {
        "no signal"
    }
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.iter().sum::<f32>() / v.len() as f32
}

/// The stability gate, reading inside the generated thought stream.
///
/// The prefill-hinge run failed at AUC ≈ 0.50 because a Gemma 4 IT model has committed to
/// nothing but "open a thought block" at that moment. This decodes into the thought and
/// scores the same pre-registered gate against the same null at several content depths.
///
/// `N*` is the **smallest** depth that clears the bar — the earliest subject commit, not the
/// deepest essay. N=2 is kept as an expected-fail control: if it passes, the anchor is
/// probably still sitting in format tokens.
#[allow(clippy::too_many_arguments)]
fn cmd_stability_decode(
    args: &ModelArgs,
    corpus_path: &PathBuf,
    layers_spec: &str,
    depths: &[usize],
    top_dims: usize,
    top_k: usize,
    max_null: usize,
    max_seq_len: usize,
    max_steps: usize,
    emit: Option<&std::path::Path>,
) -> Result<()> {
    use gguf_hooks::jacobian::DimSignature;
    use jlens_gguf::stability::{self, LayerScores, AUC_BAR, RATIO_BAR};

    let raw = std::fs::read_to_string(corpus_path)
        .with_context(|| format!("reading corpus {}", corpus_path.display()))?;
    let corpus: stability::Corpus = serde_json::from_str(&raw)?;
    if corpus.len() < 2 {
        bail!("the null population needs at least 2 subjects");
    }
    let mut depths = depths.to_vec();
    depths.sort_unstable();
    depths.dedup();

    let (mut model, tokenizer, device) = load(args)?;
    let n_layers = model.n_layers();
    let d_model = model.token_embeddings().dim(1)?;
    let layers = parse_layers(layers_spec, n_layers)?;
    // Gemma 4 IT end-of-turn set, from `main.rs::generation_eos_token_ids`.
    let eos: Vec<u32> = vec![1, 106, 50];

    let total: usize = corpus.values().map(Vec::len).sum();
    println!(
        "\nstability gate (decode-time): {} subjects, {total} prompts",
        corpus.len()
    );
    println!("layers {layers:?}, content depths {depths:?}, greedy decode\n");

    // raw[(layer, depth)][subject] = one residual per paraphrase
    type Bag = BTreeMap<(usize, usize), BTreeMap<String, Vec<Vec<f32>>>>;
    let mut bag: Bag = BTreeMap::new();
    let mut anchors = Vec::new();
    let mut samples = Vec::new();
    let mut determinism: Option<(Vec<f32>, Vec<f32>)> = None;
    let mut done = 0usize;
    let mut skipped = 0usize;

    for (subject, paraphrases) in &corpus {
        for (idx, prompt) in paraphrases.iter().enumerate() {
            let ids = jlens_gguf::tokens::encode_prompt(&tokenizer, prompt, max_seq_len)?;
            // If the corpus already ends each prompt with the thought header, the thought
            // stream starts immediately and there is no header to skip.
            // The thought stream is already open when the prompt ends with the channel
            // header, and also when the model family has no thought channel at all
            // (Gemma 3 answers directly after `<start_of_turn>model`). Only a Gemma 4
            // prompt that stops before the header has a header left to skip.
            let header_prefilled =
                prompt.trim_end().ends_with("<channel|>") || !prompt.contains("channel");

            let repeats = if done == 0 { 2 } else { 1 };
            let mut repeated: Vec<Vec<f32>> = Vec::new();

            for run in 0..repeats {
                let captured = jlens_gguf::decode::decode_and_capture(
                    &mut model,
                    &tokenizer,
                    &ids,
                    &layers,
                    &depths,
                    max_steps,
                    &eos,
                    header_prefilled,
                    &device,
                )?;
                if run > 0 {
                    if let Some(r) = captured.residuals.get(&(layers[0], depths[0])) {
                        repeated.push(r.to_vec1::<f32>()?);
                    }
                    continue;
                }
                if captured.anchor_step.is_none() {
                    skipped += 1;
                    eprintln!("  no thought-stream anchor for {subject}[{idx}]");
                    continue;
                }
                anchors.push(captured.anchor_step.unwrap_or(0));
                if samples.len() < 3 {
                    samples.push(format!(
                        "  {subject}[{idx}] anchor@step{} -> {:?}",
                        captured.anchor_step.unwrap_or(0),
                        captured.text.chars().take(90).collect::<String>()
                    ));
                }
                if let Some(r) = captured.residuals.get(&(layers[0], depths[0])) {
                    repeated.push(r.to_vec1::<f32>()?);
                }
                for (&(layer, depth), residual) in &captured.residuals {
                    bag.entry((layer, depth))
                        .or_default()
                        .entry(subject.clone())
                        .or_default()
                        .push(residual.to_vec1::<f32>()?);
                }
            }
            if repeats == 2 && repeated.len() == 2 {
                determinism = Some((repeated[0].clone(), repeated[1].clone()));
            }

            done += 1;
            if done % 20 == 0 {
                println!("  {done}/{total} prompts");
            }
        }
    }

    for line in &samples {
        println!("\n{line}");
    }
    if !anchors.is_empty() {
        let mean = anchors.iter().sum::<usize>() as f32 / anchors.len() as f32;
        println!("\nanchor step: mean {mean:.1} over {} prompts ({skipped} unanchored)", anchors.len());
    }

    let deterministic = match &determinism {
        Some((a, b)) => {
            let same = a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == y);
            println!("criterion 3 — determinism: greedy decode twice -> {}", if same { "identical" } else { "DIFFERENT" });
            same
        }
        None => {
            println!("criterion 3 — determinism: NOT MEASURED");
            false
        }
    };

    println!(
        "\n{:>6} {:>6} {:>7} {:>7} {:>10} {:>10} {:>8} {:>8}",
        "depth", "layer", "n_pos", "n_null", "med(pos)", "med(null)", "ratio", "AUC"
    );
    let mut passing: Vec<(usize, usize, f32)> = Vec::new();
    let mut emitted = Vec::new();

    for &depth in &depths {
        for &layer in &layers {
            let Some(per_subject) = bag.get(&(layer, depth)) else {
                continue;
            };
            // In-run baseline: mean/σ over this exact (layer, depth) across the corpus, so
            // it is position-matched by construction. Applied identically to every sample,
            // so it cannot manufacture separation between the populations.
            let all: Vec<&Vec<f32>> = per_subject.values().flatten().collect();
            if all.len() < 4 {
                continue;
            }
            let n = all.len() as f32;
            let mut mean = vec![0f32; d_model];
            for v in &all {
                for (m, x) in mean.iter_mut().zip(v.iter()) {
                    *m += x / n;
                }
            }
            let mut var = vec![0f32; d_model];
            for v in &all {
                for ((s, x), m) in var.iter_mut().zip(v.iter()).zip(mean.iter()) {
                    *s += (x - m) * (x - m) / (n - 1.0);
                }
            }
            let sd: Vec<f32> = var
                .iter()
                .map(|v| {
                    let s = v.sqrt();
                    if s.is_finite() && s > 1e-12 {
                        s
                    } else {
                        1.0
                    }
                })
                .collect();

            let signatures: BTreeMap<String, Vec<DimSignature>> = per_subject
                .iter()
                .map(|(subject, rows)| {
                    let sigs = rows
                        .iter()
                        .map(|row| {
                            let mut dims: Vec<(usize, f32)> = row
                                .iter()
                                .zip(&mean)
                                .zip(&sd)
                                .enumerate()
                                .map(|(i, ((x, m), s))| (i, ((x - m) / s).abs()))
                                .filter(|(_, z)| z.is_finite() && *z > 0.0)
                                .collect();
                            dims.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            dims.truncate(top_dims);
                            DimSignature::new(dims)
                        })
                        .collect();
                    (subject.clone(), sigs)
                })
                .collect();

            let scores = stability::score_layer(layer, &signatures, max_null);
            let s = scores.summarize();
            let pass = s.auc >= AUC_BAR && s.ratio() >= RATIO_BAR;
            if pass {
                passing.push((depth, layer, s.auc));
            }
            println!(
                "N={:<4} L{:<5} {:>7} {:>7} {:>10.4} {:>10.4} {:>8.2} {:>8.4}{}",
                depth,
                layer,
                s.n_positive,
                s.n_null,
                s.median_positive,
                s.median_null,
                s.ratio(),
                s.auc,
                if pass { "  PASS" } else { "" }
            );
            emitted.push(serde_json::json!({
                "depth": depth, "layer": layer, "auc": s.auc,
                "median_positive": s.median_positive, "median_null": s.median_null,
            }));

            // Dense cosine on the z-scored residual: is the signal present at all,
            // independent of how the top-k key compresses it?
            let mut positive = Vec::new();
            let mut null = Vec::new();
            let z = |row: &Vec<f32>| -> Vec<f32> {
                row.iter()
                    .zip(&mean)
                    .zip(&sd)
                    .map(|((x, m), s)| (x - m) / s)
                    .collect()
            };
            let subjects: Vec<&String> = per_subject.keys().collect();
            for sub in &subjects {
                let rows: Vec<Vec<f32>> = per_subject[*sub].iter().map(z).collect();
                for i in 0..rows.len() {
                    for j in (i + 1)..rows.len() {
                        positive.push(cosine(&rows[i], &rows[j]));
                    }
                }
            }
            for a in 0..subjects.len() {
                for b in (a + 1)..subjects.len() {
                    for x in per_subject[subjects[a]].iter().map(z) {
                        for y in per_subject[subjects[b]].iter().map(z) {
                            null.push(cosine(&x, &y));
                        }
                    }
                }
            }
            let dense = LayerScores { layer, positive, null }.summarize();
            println!(
                "{:>13} dense cosine  med(pos)={:.4} med(null)={:.4} AUC={:.4}",
                "", dense.median_positive, dense.median_null, dense.auc
            );
        }
    }

    if let Some(path) = emit {
        std::fs::write(path, serde_json::to_string_pretty(&emitted)?)?;
        println!("\nscores -> {}", path.display());
    }

    println!(
        "\nthresholds: AUC >= {AUC_BAR}, median ratio >= {RATIO_BAR}x, determinism == identical"
    );
    if !deterministic {
        println!("\nSTABILITY GATE FAIL — decode is not deterministic; nothing else counts.");
        bail!("non-deterministic decode");
    }
    match passing.first() {
        Some(&(depth, layer, auc)) => {
            println!(
                "\nSTABILITY GATE PASS — N* = {depth} (layer L{layer}, AUC {auc:.4}).\n\
                 Smallest depth that clears the bar: the earliest subject commit.\n\
                 That becomes `commit` for MultiKeyAddress; answer capture stays separate."
            );
            if depth <= 2 {
                println!(
                    "\nCAUTION: N*=2 was the expected-fail control. Passing there suggests the\n\
                     anchor is still in format tokens — check the sample generations above."
                );
            }
            Ok(())
        }
        None => {
            println!(
                "\nSTABILITY GATE FAIL — no depth clears the bar at any layer.\n\
                 Per the agreed ordering: report and stop. Redesign the commit rule;\n\
                 do not start dequant/fit."
            );
            bail!("stability gate failed at every depth")
        }
    }
}

/// The stability gate. Thresholds are pre-registered in `docs/jlens-gguf/STABILITY_GATE.md`.
#[allow(clippy::too_many_arguments)]
fn cmd_stability(
    args: &ModelArgs,
    corpus_path: &PathBuf,
    baseline_path: &PathBuf,
    layers_spec: &str,
    top_dims: usize,
    top_k: usize,
    max_null: usize,
    max_seq_len: usize,
    commit_offset: usize,
    emit: Option<&std::path::Path>,
) -> Result<()> {
    use gguf_hooks::hooks::LayerHook;
    use gguf_hooks::jacobian::DimSignature;
    use jlens_gguf::stability::{self, Verdict, AUC_BAR, PREREGISTERED_LAYER, RATIO_BAR};

    let raw = std::fs::read_to_string(corpus_path)
        .with_context(|| format!("reading corpus {}", corpus_path.display()))?;
    let corpus: stability::Corpus = serde_json::from_str(&raw)
        .with_context(|| format!("parsing {} as {{subject: [paraphrase]}}", corpus_path.display()))?;
    if corpus.len() < 2 {
        bail!("the null population needs at least 2 subjects, got {}", corpus.len());
    }

    let (mut model, tokenizer, device) = load(args)?;
    let n_layers = model.n_layers();
    let d_model = model.token_embeddings().dim(1)?;
    let layers = parse_layers(layers_spec, n_layers)?;
    let baseline = jlens_gguf::baseline::Baseline::load(baseline_path)?;
    for &layer in &layers {
        if !baseline.layers().contains(&layer) {
            bail!(
                "baseline covers {:?} but layer {layer} was requested",
                baseline.layers()
            );
        }
    }
    let sites: Vec<Site> = layers.iter().map(|&l| Site::block_out(l)).collect();

    let total: usize = corpus.values().map(Vec::len).sum();
    println!(
        "\nstability gate: {} subjects, {total} prompts, layers {:?}",
        corpus.len(),
        layers
    );
    println!(
        "commit position = last prompt token - {commit_offset} (pre-first-generated)\n"
    );

    // signatures[layer][subject] = one signature per paraphrase
    let mut signatures: BTreeMap<usize, BTreeMap<String, Vec<DimSignature>>> = BTreeMap::new();
    let mut bridges: BTreeMap<usize, BTreeMap<String, Vec<Vec<usize>>>> = BTreeMap::new();
    // Full z-scored residuals, to separate "is there signal" from "does the top-k
    // compression keep it". A sparse top-k dim set is a lossy, order-destroying hash;
    // near-tied z-scores reshuffle the winners between paraphrases.
    let mut dense: BTreeMap<usize, BTreeMap<String, Vec<Vec<f32>>>> = BTreeMap::new();
    let mut determinism_probe: Option<(DimSignature, DimSignature)> = None;
    let mut done = 0usize;

    for (subject, paraphrases) in &corpus {
        for (idx, prompt) in paraphrases.iter().enumerate() {
            // Criterion 3: run the very first prompt twice and require an exact match.
            let repeats = if done == 0 { 2 } else { 1 };
            let mut repeated: Vec<DimSignature> = Vec::new();

            for _ in 0..repeats {
                let encoded = tokenizer
                    .encode(prompt.as_str(), true)
                    .map_err(|e| anyhow::anyhow!("tokenizing {subject}[{idx}]: {e}"))?;
                let mut ids = encoded.get_ids().to_vec();
                ids.truncate(max_seq_len);
                if ids.is_empty() {
                    bail!("{subject}[{idx}] tokenised to nothing");
                }
                if ids.len() <= commit_offset {
                    bail!(
                        "{subject}[{idx}] has {} tokens, too short for --commit-offset {commit_offset}",
                        ids.len()
                    );
                }
                let commit = ids.len() - 1 - commit_offset;
                let tokens = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;

                let mut hook = jlens_gguf::probe::CaptureHook::new(sites.clone());
                model.clear_kv_cache();
                model.forward_with_hidden_hooked(
                    &tokens,
                    0,
                    Some(&mut hook as &mut dyn LayerHook),
                )?;

                for &layer in &layers {
                    let h = hook
                        .take(Site::block_out(layer))
                        .ok_or_else(|| anyhow::anyhow!("nothing captured at layer {layer}"))?;
                    let residual = h
                        .narrow(0, 0, 1)?
                        .narrow(1, commit, 1)?
                        .flatten_all()?
                        .to_dtype(DType::F32)?;
                    let ranked = baseline.standardize(&residual, layer, &device)?;
                    // Raw for the unembed — see the note in cmd_readout.
                    let logits = model.unembed(&residual.reshape((1, d_model))?)?;
                    let readout =
                        keys::read_out(&ranked, &logits, &tokenizer, top_k, top_dims)?;

                    let signature =
                        DimSignature::from_top_dimensions(&readout.top_dims, top_dims);
                    signatures
                        .entry(layer)
                        .or_default()
                        .entry(subject.clone())
                        .or_default()
                        .push(signature.clone());
                    bridges
                        .entry(layer)
                        .or_default()
                        .entry(subject.clone())
                        .or_default()
                        .push(readout.tokens.iter().map(|(id, _, _)| *id as usize).collect());
                    dense
                        .entry(layer)
                        .or_default()
                        .entry(subject.clone())
                        .or_default()
                        .push(ranked.to_vec1::<f32>()?);

                    if layer == PREREGISTERED_LAYER {
                        repeated.push(signature);
                    }
                }
            }

            if repeats == 2 {
                // Keep only one copy of the duplicated prompt in the populations.
                for &layer in &layers {
                    if let Some(v) = signatures
                        .get_mut(&layer)
                        .and_then(|m| m.get_mut(subject))
                    {
                        v.pop();
                    }
                    if let Some(v) = bridges.get_mut(&layer).and_then(|m| m.get_mut(subject)) {
                        v.pop();
                    }
                    if let Some(v) = dense.get_mut(&layer).and_then(|m| m.get_mut(subject)) {
                        v.pop();
                    }
                }
                if repeated.len() == 2 {
                    determinism_probe = Some((repeated[0].clone(), repeated[1].clone()));
                }
            }

            done += 1;
            if done % 20 == 0 {
                println!("  {done}/{total} prompts");
            }
        }
    }

    // Criterion 3.
    let deterministic = match &determinism_probe {
        Some((a, b)) => {
            let score = gguf_hooks::jacobian::weighted_jaccard(a, b);
            println!("\ncriterion 3 — determinism: same prompt twice -> {score:.6}");
            (score - 1.0).abs() < 1e-6
        }
        None => {
            println!("\ncriterion 3 — determinism: NOT MEASURED (L{PREREGISTERED_LAYER} absent)");
            false
        }
    };

    println!(
        "\n{:>6} {:>7} {:>7} {:>10} {:>10} {:>8} {:>8}",
        "layer", "n_pos", "n_null", "med(pos)", "med(null)", "ratio", "AUC"
    );
    let mut summaries = Vec::new();
    let mut emitted = Vec::new();
    for &layer in &layers {
        let per_subject = signatures.get(&layer).cloned().unwrap_or_default();
        let scores = stability::score_layer(layer, &per_subject, max_null);
        let s = scores.summarize();
        let mark = if s.auc >= AUC_BAR && s.ratio() >= RATIO_BAR {
            "  PASS"
        } else {
            ""
        };
        println!(
            "L{:<5} {:>7} {:>7} {:>10.4} {:>10.4} {:>8.2} {:>8.4}{mark}",
            s.layer, s.n_positive, s.n_null, s.median_positive, s.median_null, s.ratio(), s.auc
        );
        emitted.push(serde_json::json!({
            "layer": layer,
            "positive": scores.positive,
            "null": scores.null,
            "auc": s.auc,
            "median_positive": s.median_positive,
            "median_null": s.median_null,
        }));
        summaries.push(s);
    }

    // text_bridge, reported only — no pass/fail, per the protocol.
    println!("\ntext_bridge set-Jaccard (diagnostic, no threshold):");
    println!("{:>6} {:>10} {:>10} {:>8}", "layer", "med(pos)", "med(null)", "AUC");
    for &layer in &layers {
        let per_subject = bridges.get(&layer).cloned().unwrap_or_default();
        let as_sigs: BTreeMap<String, Vec<DimSignature>> = per_subject
            .into_iter()
            .map(|(subject, rows)| {
                let sigs = rows
                    .into_iter()
                    .map(|ids| {
                        DimSignature::new(ids.into_iter().map(|id| (id, 1.0)).collect())
                    })
                    .collect();
                (subject, sigs)
            })
            .collect();
        let s = stability::score_layer(layer, &as_sigs, max_null).summarize();
        println!(
            "L{:<5} {:>10.4} {:>10.4} {:>8.4}",
            s.layer, s.median_positive, s.median_null, s.auc
        );
    }

    // Is the signal there at all, independent of how the key compresses it?
    println!("\ndense cosine on the full z-scored residual (diagnostic, no threshold):");
    println!("{:>6} {:>10} {:>10} {:>8}", "layer", "med(pos)", "med(null)", "AUC");
    for &layer in &layers {
        let per_subject = dense.get(&layer).cloned().unwrap_or_default();
        let subjects: Vec<&String> = per_subject.keys().collect();
        let mut positive = Vec::new();
        let mut null = Vec::new();
        for s in &subjects {
            let v = &per_subject[*s];
            for i in 0..v.len() {
                for j in (i + 1)..v.len() {
                    positive.push(cosine(&v[i], &v[j]));
                }
            }
        }
        for a in 0..subjects.len() {
            for b in (a + 1)..subjects.len() {
                for x in &per_subject[subjects[a]] {
                    for y in &per_subject[subjects[b]] {
                        null.push(cosine(x, y));
                    }
                }
            }
        }
        let scores = jlens_gguf::stability::LayerScores { layer, positive, null };
        let s = scores.summarize();
        println!(
            "L{:<5} {:>10.4} {:>10.4} {:>8.4}",
            s.layer, s.median_positive, s.median_null, s.auc
        );
    }

    if let Some(path) = emit {
        std::fs::write(path, serde_json::to_string_pretty(&emitted)?)?;
        println!("\nper-pair scores -> {}", path.display());
    }

    let verdict = stability::verdict(&summaries, deterministic);
    println!(
        "\nthresholds: AUC >= {AUC_BAR}, median ratio >= {RATIO_BAR}x, determinism == 1.000, \
         pre-registered layer L{PREREGISTERED_LAYER}"
    );
    match verdict {
        Verdict::Pass => {
            println!(
                "\nSTABILITY GATE PASS — keys are re-hittable at the pre-registered layer.\n\
                 dim_signature is usable for within-model clustering."
            );
            Ok(())
        }
        Verdict::PartialPass => {
            println!(
                "\nSTABILITY GATE PARTIAL PASS — some layer clears the bar but L{PREREGISTERED_LAYER} does not.\n\
                 The approach holds; the layer choice was over-fit to the 3-prompt preview.\n\
                 Re-register the winning layer and re-run before building on it."
            );
            Ok(())
        }
        Verdict::Fail => {
            println!(
                "\nSTABILITY GATE FAIL — keys are not re-hittable at any layer tested.\n\
                 Diagnose before touching the fitter. See the failure modes in\n\
                 docs/jlens-gguf/STABILITY_GATE.md."
            );
            bail!("stability gate failed")
        }
    }
}

/// Collect per-layer residual mean/sd over a corpus.
#[allow(clippy::too_many_arguments)]
fn cmd_baseline(
    args: &ModelArgs,
    prompts_path: &PathBuf,
    layers_spec: &str,
    max_seq_len: usize,
    commit_only: bool,
    commit_offset: usize,
    out: &PathBuf,
) -> Result<()> {
    use gguf_hooks::hooks::LayerHook;

    let prompts = read_prompts(prompts_path)?;
    let (mut model, tokenizer, device) = load(args)?;
    let n_layers = model.n_layers();
    let d_model = model.token_embeddings().dim(1)?;
    let layers = parse_layers(layers_spec, n_layers)?;
    let sites: Vec<Site> = layers.iter().map(|&l| Site::block_out(l)).collect();

    let mut acc = jlens_gguf::baseline::Accumulator::new(d_model);
    let mut used = 0usize;
    for (i, prompt) in prompts.iter().enumerate() {
        let ids = match jlens_gguf::tokens::encode_prompt(&tokenizer, prompt, max_seq_len) {
            Ok(ids) => ids,
            Err(_) => continue,
        };
        let seq_len = ids.len();
        // Same exclusion as the fit: attention-sink positions have atypical statistics and
        // would drag the mean toward structure no real readout position ever sees.
        let valid: Vec<usize> = if commit_only {
            if seq_len <= commit_offset {
                continue;
            }
            vec![seq_len - 1 - commit_offset]
        } else {
            match fit::valid_positions(seq_len, fit::SKIP_FIRST_N_POSITIONS) {
                Ok(v) => v,
                Err(_) => (0..seq_len).collect(),
            }
        };
        let tokens = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
        let idx = Tensor::new(
            valid.iter().map(|&p| p as u32).collect::<Vec<_>>().as_slice(),
            &device,
        )?;

        let mut hook = jlens_gguf::probe::CaptureHook::new(sites.clone());
        model.clear_kv_cache();
        model.forward_with_hidden_hooked(&tokens, 0, Some(&mut hook as &mut dyn LayerHook))?;

        for &layer in &layers {
            let h = hook
                .take(Site::block_out(layer))
                .ok_or_else(|| anyhow::anyhow!("no activation captured at layer {layer}"))?;
            let rows: Vec<f32> = h
                .narrow(0, 0, 1)?
                .index_select(&idx, 1)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            acc.push(layer, &rows)?;
        }
        used += 1;
        if used % 10 == 0 {
            println!("  {used}/{} prompts", prompts.len());
        }
    }

    let baseline = acc.finish()?;
    baseline.save(out)?;
    println!(
        "\nwrote {} — {} layers, {} positions from {used} prompts",
        out.display(),
        baseline.layers().len(),
        baseline.n_samples
    );
    Ok(())
}

/// Logit-lens telemetry: capture, unembed, emit. No fit, no differencing.
#[allow(clippy::too_many_arguments)]
fn cmd_readout(
    args: &ModelArgs,
    prompt: &str,
    layers_spec: &str,
    positions_spec: &str,
    top_k: usize,
    top_dims: usize,
    emit: Option<&std::path::Path>,
    state_dir: Option<&std::path::Path>,
    tag: Option<String>,
    max_seq_len: usize,
    baseline_path: Option<&std::path::Path>,
    verify: bool,
) -> Result<()> {
    use gguf_hooks::hooks::LayerHook;
    use std::io::Write;

    let (mut model, tokenizer, device) = load(args)?;
    let n_layers = model.n_layers();
    let d_model = model.token_embeddings().dim(1)?;
    let layers = parse_layers(layers_spec, n_layers)?;
    let source = model.variant_name().to_string();
    let baseline = baseline_path
        .map(jlens_gguf::baseline::Baseline::load)
        .transpose()?;
    if let Some(b) = &baseline {
        for &layer in &layers {
            if !b.layers().contains(&layer) {
                bail!(
                    "baseline covers layers {:?} but layer {layer} was requested; \
                     re-run `baseline` with matching --layers",
                    b.layers()
                );
            }
        }
    } else {
        eprintln!(
            "warning: no --baseline. dim_signature will rank the model's constant outlier\n\
             dimensions, which are the same for every prompt, so the keys will not\n\
             discriminate. Run `jlens-gguf baseline` first."
        );
    }

    let ids = jlens_gguf::tokens::encode_prompt(&tokenizer, prompt, max_seq_len)?;
    let seq_len = ids.len();
    let tokens = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;

    let wanted: Vec<usize> = positions_spec
        .split(',')
        .map(|s| {
            let p: i64 = s.trim().parse().context("position must be an integer")?;
            let resolved = if p < 0 { seq_len as i64 + p } else { p };
            if resolved < 0 || resolved >= seq_len as i64 {
                bail!("position {p} out of range for a {seq_len}-token prompt");
            }
            Ok(resolved as usize)
        })
        .collect::<Result<Vec<_>>>()?;

    // One prefill captures every requested layer at once.
    let sites: Vec<Site> = layers.iter().map(|&l| Site::block_out(l)).collect();
    let mut hook = jlens_gguf::probe::CaptureHook::new(sites);
    model.clear_kv_cache();
    model.forward_with_hidden_hooked(&tokens, 0, Some(&mut hook as &mut dyn LayerHook))?;

    if verify {
        // The model's own logits for the same tokens, straight through `forward()`.
        model.clear_kv_cache();
        let logits = model.forward(&tokens, 0)?.to_dtype(DType::F32)?;
        let vocab = logits.dim(logits.rank() - 1)?;
        let flat = logits.reshape(((), vocab))?;
        let last = flat.narrow(0, flat.dim(0)? - 1, 1)?;
        let readout = keys::read_out(&last, &last, &tokenizer, top_k, 4)?;
        let words: Vec<String> = readout
            .tokens
            .iter()
            .map(|(_, text, score)| format!("{text:?}({score:.2})"))
            .collect();
        eprintln!("\nverify: model.forward() top-{top_k} = {}", words.join(" "));
        eprintln!(
            "        the L{} record below must match this exactly.\n",
            n_layers - 1
        );
    }

    let mut sink: Box<dyn Write> = match emit {
        Some(path) => {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            Box::new(std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .with_context(|| format!("opening {}", path.display()))?,
            ))
        }
        None => Box::new(std::io::stdout()),
    };

    let mut written = 0usize;
    for &layer in &layers {
        let captured = hook
            .take(Site::block_out(layer))
            .ok_or_else(|| anyhow::anyhow!("no activation captured at layer {layer}"))?;
        for &position in &wanted {
            // [d_model] — the residual as it stood at this layer and position.
            let residual = captured
                .narrow(0, 0, 1)?
                .narrow(1, position, 1)?
                .flatten_all()?
                .to_dtype(DType::F32)?;
            // Two different treatments, for the reason set out in `baseline.rs`:
            //   text_bridge  <- unembed(h - μ)      centring only; basis-preserving
            //   dim_signature <- rank |(h - μ)/σ|   z-score; a ranking space, not a basis
            // Without a baseline both fall back to the raw residual, and the key then ranks
            // the model's constant outlier dimensions instead of the thought.
            // Measured 2026-08-02: centring before the unembed destroys the readout.
            // On forced-completion prompts whose paraphrases must predict the same word,
            // centred top-8 token sets shared *nothing* (median Jaccard 0.0000) — μ carries
            // the component driving the shared prediction, so h-μ removes the answer. The
            // un-centred readout, by contrast, reproduces model.forward() exactly.
            //   text_bridge  <- unembed(h)          raw; this is the logit lens
            //   dim_signature <- rank |(h-μ)/σ|     z-score; ranking only
            let for_unembed = residual.clone();
            let for_ranking = match &baseline {
                Some(b) => b.standardize(&residual, layer, &device)?,
                None => residual.clone(),
            };
            // Logit lens: unembed with no transport. `unembed` applies the final norm, so
            // the residual is read in the basis the model decodes with — noisy at low
            // layers by construction, not by mistake.
            let logits = model.unembed(&for_unembed.reshape((1, d_model))?)?;
            let readout = keys::read_out(&for_ranking, &logits, &tokenizer, top_k, top_dims)?;

            let token_id = ids[position];
            let token = tokenizer
                .decode(&[token_id], false)
                .unwrap_or_else(|_| format!("<{token_id}>"));

            let mut rec = jlens_gguf::telemetry::record(
                &readout,
                jlens_gguf::telemetry::LensKind::Logit,
                &source,
                &args.model,
                layer,
                position,
                n_layers,
                d_model,
                token_id,
                token,
                tag.clone(),
            );
            if let Some(dir) = state_dir {
                rec.state_ref = Some(jlens_gguf::telemetry::save_state(&residual, dir)?);
            }
            writeln!(sink, "{}", serde_json::to_string(&rec)?)?;
            written += 1;
        }
    }
    sink.flush()?;

    if let Some(path) = emit {
        eprintln!(
            "wrote {written} records to {} (lens=logit, source={source})",
            path.display()
        );
    }
    Ok(())
}

fn cmd_apply(
    args: &ModelArgs,
    lens_path: &PathBuf,
    prompt: &str,
    positions_spec: &str,
    band_spec: &str,
    top_k: usize,
) -> Result<()> {
    let lens = Lens::load(lens_path)?;
    let (mut model, tokenizer, device) = load(args)?;
    let band = BandId::named(band_spec);
    let positions: Vec<i64> = positions_spec
        .split(',')
        .map(|s| s.trim().parse::<i64>().context("position must be an integer"))
        .collect::<Result<Vec<_>>>()?;

    let layers = lens.source_layers();
    let (lens_logits, model_logits, ids) = lens.apply(
        &mut model,
        &tokenizer,
        prompt,
        &layers,
        Some(&positions),
        &band,
        &device,
    )?;

    println!("\nprompt   {prompt:?}");
    println!("tokens   {} (band {band})", ids.len());
    for (i, &p) in positions.iter().enumerate() {
        println!("\nposition {p}:");
        for (&layer, logits) in lens_logits.iter() {
            let row = logits.narrow(0, i, 1)?;
            let readout = keys::read_out(&row, &row, &tokenizer, top_k, 8)?;
            let words: Vec<String> = readout
                .tokens
                .iter()
                .map(|(_, text, score)| format!("{text:?}({score:.2})"))
                .collect();
            println!("  L{layer:<3} {}", words.join(" "));
        }
        let row = model_logits.narrow(0, i, 1)?;
        let readout = keys::read_out(&row, &row, &tokenizer, top_k, 8)?;
        let words: Vec<String> = readout
            .tokens
            .iter()
            .map(|(_, text, score)| format!("{text:?}({score:.2})"))
            .collect();
        println!("  model {}", words.join(" "));
    }
    Ok(())
}

fn cmd_sweep(
    args: &ModelArgs,
    prompt: &str,
    layer: usize,
    eps_spec: &str,
    probes: usize,
    max_seq_len: usize,
) -> Result<()> {
    let eps_values: Vec<f32> = eps_spec
        .split(',')
        .map(|s| s.trim().parse::<f32>().context("eps must be a float"))
        .collect::<Result<Vec<_>>>()?;
    let (mut model, tokenizer, device) = load(args)?;
    let n_layers = model.n_layers();
    if layer >= n_layers {
        bail!("layer {layer} out of range for a {n_layers}-layer model");
    }
    let d_model = model.token_embeddings().dim(1)?;

    let (tokens, seq_len, valid, target_idx) =
        prepare_prompt(&tokenizer, prompt, max_seq_len, &device)?;
    let mask = band_mask(seq_len, &valid, &device)?;
    let source = Site::block_out(layer);
    let target = Site::block_out(n_layers - 1);

    let rms = source_rms(&mut model, &tokens, source, &valid, &device)?;
    let target_rms = source_rms(&mut model, &tokens, target, &valid, &device)?;
    let rows = jlens_gguf::basis::random_basis(d_model, probes.max(1), 7)?;
    let dirs_basis = Basis::from_rows(BasisKind::Random, d_model, rows)?;
    let dirs = dirs_basis.chunk(0, dirs_basis.rank(), &device)?;

    // f32 carries ~7 decimal digits, so each target element arrives with roughly
    // |h|·2^-24 of accumulated error, and summing n of them incoherently grows it by √n.
    // Any Δh below this floor is rounding, not response — which is what makes the
    // rounding branch show up as ||J v|| ∝ 1/ε.
    let n_summed = (valid.len() * d_model) as f32;
    let noise_floor = target_rms * f32::EPSILON * n_summed.sqrt();

    println!("\nlayer {layer} -> {} | source RMS {rms:.4}", n_layers - 1);
    println!("target RMS {target_rms:.1}  ->  f32 cancellation floor ~{noise_floor:.2}");
    println!("{} probes, {} valid positions\n", dirs_basis.rank(), valid.len());
    println!(
        "{:>10}  {:>12}  {:>10}  {:>12}  {:>9}",
        "eps_rel", "eps_abs", "||J v||", "||sum dh||", "cos(prev)"
    );

    let mut previous: Option<Vec<f32>> = None;
    let mut plateau: Vec<(f32, f32)> = Vec::new();
    for &eps_rel in &eps_values {
        let eps = eps_rel * rms;
        let cols = fit::probe_columns(
            &mut model,
            &tokens,
            &dirs,
            eps,
            source,
            target,
            &mask,
            valid.len(),
            &target_idx,
        )?;
        let flat: Vec<f32> = cols.flatten_all()?.to_vec1()?;
        let norm = flat.iter().map(|v| v * v).sum::<f32>().sqrt();
        // Undo the estimator's scaling to recover the raw summed difference. Constant
        // here across ε is the tell-tale of a rounding-dominated measurement.
        let raw = norm * 2.0 * eps * valid.len() as f32;
        let cos = previous
            .as_ref()
            .map(|prev| cosine(prev, &flat))
            .unwrap_or(f32::NAN);
        println!("{eps_rel:>10.1e}  {eps:>12.5}  {norm:>10.4}  {raw:>12.3}  {cos:>9.5}");
        plateau.push((eps_rel, cos));
        previous = Some(flat);
    }

    // A plateau means consecutive ε agree in *direction*: the estimator is measuring the
    // same linear map, not noise (too small) or curvature (too large).
    let stable: Vec<f32> = plateau
        .iter()
        .filter(|(_, cos)| cos.is_finite() && *cos > 0.999)
        .map(|(eps, _)| *eps)
        .collect();
    println!();
    if stable.len() >= 2 {
        println!(
            "PLATEAU: cos > 0.999 across eps_rel {:.1e} .. {:.1e} — fit in that range.",
            stable.first().unwrap(),
            stable.last().unwrap()
        );
    } else {
        println!(
            "NO PLATEAU: no run of consecutive eps agrees to cos > 0.999.\n\
             The forward-mode estimator is not trustworthy at this quantisation for this\n\
             layer. This is a negative result — do not pick the prettiest eps and proceed."
        );
    }
    Ok(())
}

fn cmd_identity(
    args: &ModelArgs,
    prompt: &str,
    probes: usize,
    eps_rel: f32,
    max_seq_len: usize,
) -> Result<()> {
    let (mut model, tokenizer, device) = load(args)?;
    let n_layers = model.n_layers();
    let d_model = model.token_embeddings().dim(1)?;
    // Source == target: the transport from a layer to itself can only be the identity.
    let layer = n_layers - 1;
    let site = Site::block_out(layer);

    let (tokens, seq_len, valid, target_idx) =
        prepare_prompt(&tokenizer, prompt, max_seq_len, &device)?;
    let mask = band_mask(seq_len, &valid, &device)?;
    let rms = source_rms(&mut model, &tokens, site, &valid, &device)?;
    let eps = eps_rel * rms;

    let probes = probes.min(d_model).max(1);
    let identity = Basis::identity(d_model);
    let dirs = identity.chunk(0, probes, &device)?;

    let cols = fit::probe_columns(
        &mut model,
        &tokens,
        &dirs,
        eps,
        site,
        site,
        &mask,
        valid.len(),
        &target_idx,
    )?;
    let cols: Vec<Vec<f32>> = cols.to_vec2()?;

    // Column j of J should be e_j: 1 at row j, 0 elsewhere.
    let mut worst_diag = f32::INFINITY;
    let mut worst_off = 0f32;
    let mut sq_error = 0f64;
    for (j, col) in cols.iter().enumerate() {
        for (r, &value) in col.iter().enumerate() {
            let expected = if r == j { 1.0 } else { 0.0 };
            sq_error += ((value - expected) as f64).powi(2);
            if r == j {
                worst_diag = worst_diag.min(value);
            } else {
                worst_off = worst_off.max(value.abs());
            }
        }
    }
    let rel = (sq_error.sqrt() / (probes as f64).sqrt()) as f32;

    println!("\nsource == target == L{layer}, {probes} columns, eps {eps:.5} ({eps_rel:.1e} rel)");
    println!("  min diagonal      {worst_diag:.6}   (want 1.0)");
    println!("  max off-diagonal  {worst_off:.6}   (want 0.0)");
    println!("  ||J - I||_F/√n    {rel:.6}");
    if rel < 1e-2 {
        println!(
            "\nGATE 1 PASS: the estimator recovers the identity transport.\n\
             Scope: this exercises the band mask, the 1/(2eps|band|) scale, the sign of the\n\
             central difference, and per-batch probe isolation. It does NOT exercise any\n\
             transformer block — source == target means the capture sees the injected value\n\
             directly, so exactness here is algebraic. Run `sweep` for propagation."
        );
        Ok(())
    } else {
        println!(
            "\nGATE 1 FAIL: J is not the identity where it provably must be.\n\
             Suspect the position mask, the sign of the central difference, or batch\n\
             aliasing in the probe hook. Nothing downstream is meaningful until this passes."
        );
        bail!("identity gate failed: ||J - I||_F/√n = {rel:.6}")
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return f32::NAN;
    }
    dot / (na * nb)
}

/// Unused today, kept because `BTreeMap` in the signature documents the layer ordering the
/// fit relies on.
#[allow(dead_code)]
fn _layer_order(bases: &BTreeMap<usize, Basis>) -> Vec<usize> {
    bases.keys().copied().collect()
}
