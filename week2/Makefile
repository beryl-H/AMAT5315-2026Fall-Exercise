.PHONY: reproduce

# Regenerate the default 100-atom LJ-fluid trajectory (md run, default flags)
# into artifacts/, as a fresh-clone reproduction.
reproduce:
	mkdir -p artifacts
	cargo run -- run --out artifacts/trajectory.txt
