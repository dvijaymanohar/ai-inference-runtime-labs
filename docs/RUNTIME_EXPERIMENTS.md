# Runtime experiments

For each runtime/provider:
- validate outputs against a trusted reference
- record provider/operator placement
- separate model-load/cold-start from steady state
- sweep batch and relevant dynamic shapes
- test supported precision modes
- record memory footprint
- identify CPU/GPU fallback
- compare end-to-end latency with kernel/device timing where available

A faster isolated kernel does not imply lower user-visible latency.
