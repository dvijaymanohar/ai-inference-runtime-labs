# Runtime labs

Run in this order:
```bash
python examples/export_and_compare.py
python examples/provider_inspection.py
python examples/cold_warm_latency.py
python examples/batch_sweep.py
python examples/quantization_dynamic.py
python benchmarks/runtime_benchmark.py
```

Learn-by-doing questions:
1. Which work belongs to model load, graph optimization, first inference, and steady state?
2. Which execution providers are actually available and used?
3. Where does batching improve throughput, and where does latency become unacceptable?
4. What numerical change does quantization introduce?
5. Does a smaller/faster model remain acceptable for the task-level quality metric?
6. With CUDA/TensorRT installed, repeat identical shapes/dtypes and document any fallback.
