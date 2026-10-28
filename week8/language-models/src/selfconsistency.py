"""Self-consistency evaluation: the model's own samples are exactly as
surprising as it predicts.

Protocol (SPEC, frozen): 128 sequences x 128 fixed-length tokens (EOS stop
ignored), sampled at T=1 (seed 531506) and T=0.7; both scored by the T=1
model. Batch 128 with a rolling 64-token prefix window (positions consistent,
initial context EOS). Before each draw the T=1 entropy is recorded; the
surprise of the drawn token is -ln P1(chosen). Per sequence
D_i = mean_t(surprise - entropy); z = mean(D) / (std(D, ddof=1)/sqrt(128)) —
the SE is over SEQUENCES, not tokens. Gates: |z| <= 4 at T=1; |z| > 40 at
T=0.7. The z formula is tested on oracle fixtures; the long run happens at
formal time on saved final weights, never on a smoke model."""
import numpy as np


def entropy_zscore(surprises, entropies):
    """surprises/entropies: [n_seq, n_tokens]. Returns the z-score of the
    per-sequence mean difference (SE over sequences)."""
    D = (surprises - entropies).mean(axis=1)
    n = len(D)
    se = D.std(ddof=1) / np.sqrt(n)
    return float(D.mean() / se)


def entropy_stats(surprises, entropies):
    D = (surprises - entropies).mean(axis=1)
    return {"mean_D": float(D.mean()),
            "mean_pred_entropy": float(entropies.mean()),
            "mean_sample_surprise": float(surprises.mean()),
            "se_over_sequences": float(D.std(ddof=1) / np.sqrt(len(D))),
            "n_sequences": int(len(D)),
            "formula": "z = mean_i(D_i) / (std_i(D_i, ddof=1)/sqrt(n_seq)); D_i = mean_t(surprise - entropy)"}


def gates(z_T1, z_T07):
    return {"T1_absz_le_4": abs(z_T1) <= 4.0, "T07_absz_gt_40": abs(z_T07) > 40.0}
