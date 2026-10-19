# Authoritative source

This exercise follows the **local** 12-page Week 7: Inverse problems and
generative priors PDF explicitly selected by the student on 2026-10-19.

- Local source: `D:/Keria/Course/AMAT5315/week7-learning-sheet.pdf`
- SHA-256: `e5183bcd8fa54bcdb8ca5952c523c95cb030058e226f8ec68a175ad50dcf80e4`
- On 2026-10-19 the live course PDF at
  <https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/week7-learning-sheet.pdf>
  instead had title **Diffusion and flow matching** and SHA-256
  `295da2ac6943137b0c6363727e8c31942b0af8ed8cd0996676fde321c1a59c5d`.
  These are different versions. No requirements are silently substituted.
- Prepared images: <https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week7-images.npz>
- The current downloaded archive SHA-256 is
  `edce7ff8b6c2cd8b134846cdab7d7555c22ce2c0da89c97364c438575b4e344c`.
  It contains four 3-channel images, not the ten MNIST digits specified by the
  local PDF, and is **not used**. `prepare_inputs.py` reproducibly reconstructs
  the specified first/second occurrences of digits 0..9 from official MNIST
  test IDX files, using the exact row IDs and preprocessing in the local PDF.
  Original archive digests and final array digests are in
  [input-provenance.json](evidence/input-provenance.json).

The existing Week 6 diffusion source and EMA checkpoint are reused read-only.
The later `week6/energy-based/` exercise is separate. No new training or cloud
instance is required. The repository baseline is
`af4bbe8622ac8e554a3961f4d7a172730802f6ca`, initially clean.

Readings: [Candes and Romberg, abstract](https://arxiv.org/abs/math/0611957),
[Lugmayr et al., RePaint, sections 4.1-4.2 and Figure 3](https://arxiv.org/html/2201.09865v4).
