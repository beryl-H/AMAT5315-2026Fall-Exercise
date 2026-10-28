# PART 4 DECODER PROTOCOL (frozen before any formal decoder training)

Frozen by pi (native WSL, volcengine-plan / glm-5-3-flash) on 2026-10-28,
per pi-task-08 TASK A. This file is the authoritative protocol for the
`formal-decoder-02` namespace. DO NOT MODIFY after the ready marker until
Codex announces all Part 4 training is done.

## Why the correction (verified defect, not post-hoc gating)

The original formal-01 used a symmetric variant: no positional embeddings,
no causal mask. Then answer logits are invariant under swapping a and b, so
an input set containing BOTH (a,b)->c1 and (b,a)->c2 with c1 != c2 is not
memorizable. The worksheet's random-label control contains exactly 393
conflicting unordered training pairs (both members inside the 30% training
split); the maximal reachable training accuracy is 0.860786 < 0.99, so the
required "memorized by step 200" behaviour is impossible. formal-01 was
STOPPED and INVALIDATED for the worksheet controls (logs/checkpoints
preserved in artifacts/part4/formal-01/INVALIDATED-PROTOCOL.json; see
week8-support/part4-symmetry-defect.json). Two completed main seeds (2, 3)
and all logs remain as preserved artifacts. This correction restores the
Part 2 decoder structure and removes ONLY LayerNorm, exactly as the sheet
says. It is a required capacity fix, not fitting to reference curves.

## Architecture (frozen)

- One layer, NO LayerNorm; width 64; 4 heads (head width 16); MLP 64->256->64
  with biases; GELU. Token embeddings 98x64 (residues 0..96 + '=' id 97) plus
  LEARNED POSITIONAL EMBEDDINGS 3x64, both init N(0, sqrt(2/(64+97))),
  biases zero; positions summed with token embeddings.
- Causal tril attention mask before softmax (Part 2 decoder structure).
  Answer read from logits at position 2 ('='), softmax over the 97 residue
  classes (chance 1/97, uniform loss ln 97).
- Total parameters: 62,497 = 62,305 + 192 (positional). No LayerNorm.
- Input sequence [a, b, '=']; the answer never appears as input.
- Init norm ~28 reported (not tuned). Float32, no clipping.

## Optimizer and budgets (unchanged from the invalidated variant)

- Full-batch AdamW, lr 3e-3, betas (0.9, 0.98), weight decay 1 (wd 0 and
  50%-fraction controls as below). State threaded through the jitted step.
- Up to 10000 steps; stop at first-logged test accuracy >= 0.99 plus 1000
  more steps (stop_step = min(10000, grok_step + 1000)).
- Log grid: step 0 (initialization, before any update) and every 100 steps:
  train/test loss, train/test accuracy, parameter norm.
- memorization step = first LOGGED step with train acc >= 0.99; grokking
  step = first LOGGED step with test acc >= 0.99; delay = grok/memorize.
  Checkpoints at init, memorization (explicit null if absent), final;
  reload-verified; completion receipt before FFT/ablation postprocessing.

## Runs (namespace formal-decoder-02; formal-01 never overwritten)

- Main: seeds 0..4, 30% split, wd 1.
- Controls: seeds 0..2 with wd 0; seeds 0..2 with 50% fraction.
- Random-label control: seed 0, main settings, 3000 steps; labels = the
  true-label VECTOR shuffled across all 9409 pair indices with
  np.random.default_rng(8970) (class frequencies preserved; NOT a class
  permutation). With the decoder this memorization control is expressible:
  the 393 conflicting unordered training pairs are memorizable (verified by
  tests/test_part4_decoder.py capacity check).

## Split and labels (unchanged)

- Split: np.random.default_rng(897) permutation of the 9409 pair indices;
  30% = round(0.30*9409) = 2823 train; 50% = round(0.50*9409) = 4704
  (nested); label-independent; train/test never overlap.

## Fourier analysis and ablations (unchanged)

- np.fft.rfft of residue rows over residue index a=0..96; power_k =
  sum_dims |X_k|^2 for k=1..48 (conjugate pairs folded); top-5 share
  denominator = total power over k=1..48 (DC excluded from the share and
  preserved in every ablation); '=' row never in DFT/ablations; top-5 from
  each snapshot's own spectrum (init, memorization, final).
- Ablations on the final model: keep-only key5, remove key5, remove 5
  random other frequencies (10 draws, np.random.default_rng(4497), from the
  complement of the final top-5), keep-only 5 random other. Exact float64
  trig reconstruction: x[a] = DC + (2/97) sum_k (C_k cos + S_k sin).

## Execution rules

- Codex manages all formal workers; pi does not launch formal training.
- Every run in an exclusive-new directory; no overwrites; all logs,
  checkpoints and receipts preserved even on failure; postprocessing
  failures never force retraining.
- Separately: the formal-01 symmetric-variant artifacts are preserved as an
  INVALIDATED protocol record and are not part of the decoder evidence.
