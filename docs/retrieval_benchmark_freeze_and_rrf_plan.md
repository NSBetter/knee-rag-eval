# Retrieval Benchmark Freeze and RRF Plan

## Freeze rule

The pilot is frozen by copying the canonical CSV to a versioned snapshot
and recording SHA-256 hashes for both the benchmark and Gold Corpus.

The frozen pilot must contain:

- 12 selected pilot rows
- 12 verified rows
- at least 2 multi-chunk rows
- at least 1 no-Gold row
- at least 1 genuine cross-source row

RET-006 remains a multi-chunk, cross-source query. Its Gold evidence must
remain independently justified; the cross-source requirement must not be
satisfied by adding merely related chunks.

## RRF baseline

The first hybrid baseline uses Reciprocal Rank Fusion:

`score(d) = sum(1 / (k + rank_i(d)))`

Configuration:

- `k = 60`
- input depth = 10 from BM25 and Dense
- output depth = 10
- evaluation uses the frozen pilot benchmark
- strict Gold metrics remain the primary metrics

Because Dense already reaches Hit@3 and Hit@5 of 1.0 on this pilot, RRF
may tie or reduce the Dense result. A neutral or negative result remains
informative and should not be hidden.
