# SOURCE — provenance for week7/diffusion-flow

- Lecture: "Week 7: Diffusion and flow matching", Jin-Guo Liu, Xi-Wei Pan.
  Text source: `/mnt/d/Keria/Course/AMAT5315/week7-support/diffusion-flow-20261021/learning-sheet.txt`
  (19 pages). PDF: `week7-diffusion-flow-20261021.pdf` in the same directory.
  PDF SHA256: `4147c45cc6ac9f6e1815ba33684a1c4ed50c87d8d89ffb14f3dccab6bcfddbd5` (verified).
- Authoritative task prompt: `pi-part1.md` in the support directory (Part 1
  only this turn; coordinator verifies per Part).
- Repository rules: `../../AGENTS.md` (AMAT5315-2026Fall-Exercise/AGENTS.md).
- Git: branch `week7-diffusion-flow-20261021`, starts at commit
  `a72f825ad17a3bc089e015d3bac02785afacaae7` ("docs: require WSL pi assistance
  for AMAT5315 coursework").
- Implementation is JAX float32 on CPU; `diffusion.toml` is transcribed
  verbatim from the lecture sheet's code block.
- Agent: native `pi` in WSL, provider/model kept as the student's existing
  Ark Coding Plan glm-5.3-flash selection. Coordinator: communicates in
  Chinese, handles final publication. No push/merge/cloud/subagents this turn.
