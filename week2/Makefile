# Week 2 Part 4: regenerate the default contract run.
# Test verification:    cargo test --manifest-path md/Cargo.toml --release
# Physics verification: cargo run --manifest-path md/Cargo.toml --release -- check artifacts
# Video (needs ffmpeg): cargo run --manifest-path md/Cargo.toml --release -- video artifacts --out artifacts/run.mp4

reproduce:
	cargo run --manifest-path md/Cargo.toml --release -- run --out artifacts

.PHONY: reproduce
