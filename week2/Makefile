.PHONY: reproduce

# Regenerate the default 100-atom LJ-fluid trajectory into artifacts/ as a
# fresh-clone reproduction. All physics parameters are spelled out: the CLI
# deliberately has no defaults for them.
reproduce:
	mkdir -p artifacts
	cargo run -- run -n 100 --temp 1.0 --dt 0.005 --steps 10000 --equil 1000 --out artifacts/trajectory.txt
