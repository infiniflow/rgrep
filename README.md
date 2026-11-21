# rgrep

Local code search for agents: a CLI plus a daemon. The body is Go; tree-sitter
splitting and HuggingFace tokenization live in a Rust static library inside the
repo; embeddings come from LateOn-Code-edge (ColBERT multi-vector) through
onnxruntime.

All retrieval runs in **Infinity**: PLAID multi-vector recall and BM25 full-text
recall are executed by the engine and fused there with `FUSION(...)`. SQLite is
kept for what a search engine cannot answer — the per-file fingerprint that
makes indexing incremental. The two are not alternatives; they have different
jobs. See [`docs/INFINITY_INTEGRATION.md`](docs/INFINITY_INTEGRATION.md).

The engine is **embedded**: it is a shared library the daemon loads, it serves
the ordinary thrift API on a unix socket in the data dir, and it stops when the
daemon does. There is no server to install, no port to pick and nothing to
supervise. `infinity_mode: "remote"` switches to a server somebody else runs.

## Architecture

```
            agent (bash / MCP)
              │  rgrep search ...        (short-lived, IPC only)
              ▼
        ┌─────────────┐   JSON lines over a unix socket
        │  rgrep CLI  │ ───────────────┐
        └─────────────┘                │
                                       ▼
       ┌───────────────────────────────────────────────────┐
       │  rgrep serve (daemon)                             │  the only writer
       │  ├ walk / hash / plan                             │
       │  ├ native.Split   (Rust)                          │  tree-sitter
       │  ├ native.Tokenize (Rust)                         │  HF tokenizer
       │  ├ embed (ONNX int8)                              │  LateOn-Code-edge
       │  └ internal/infinity ──┐                          │
       │                        │ unix socket, in-process  │
       │     ┌──────────────────▼───────────────────────┐  │
       │     │ Infinity (libinfinity_embedded)          │  │
       │     │ PLAID + FULLTEXT + FUSION                │  │
       │     └──────────────────┬───────────────────────┘  │
       └────────────────────────┼──────────────────────────┘
                                ▼
                 $DATA_DIR/infinity/{catalog,data,wal}

       catalog (SQLite): $DATA_DIR/rgrep.db — files / meta only
```

Process model: the CLI starts the daemon on demand, and the daemon exits after
30 minutes idle. `--in-process` skips the daemon entirely (CI, sandboxes) and
still gets its own embedded engine for the length of the command.

The Rust static library (`native/rgrep_native`) exposes only a small C ABI:

| C ABI | purpose |
|---|---|
| `rg_codesplit(lang, path, content, len, &err)` | tree-sitter code units as JSON |
| `rg_tokenize(path, text, len, &err)` | tokenize with the model's `tokenizer.json` |
| `rg_free`, `rg_languages`, `rg_abi_version` | helpers |

BPE is not reimplemented in Go, and Go never calls the tree-sitter C core
directly: one cgo call returns the whole structured tree, and the rest is Go.

## Quick start

```bash
# 1. Build. `make engine` compiles the embedded engine's shared library from
#    the engine checkout next to this repo and copies it into bin/lib.
make native                 # tree-sitter + tokenizer, Rust static library
make engine                 # the database itself (needs ../infinity/rust_engine)
RGREP_PROXY=socks5h://127.0.0.1:7891 make ort   # optional libonnxruntime
make build                  # produces bin/rgrep

# 2. Index and search. Nothing else has to be running.
bin/rgrep index ~/code/my-repo -y
bin/rgrep search "how are retries configured" -k 5
bin/rgrep search "parseConfig" -k 3 --json
bin/rgrep get internal/config/config.go:37
```

`bin/rgrep` finds the engine in `bin/lib`. Set `infinity_lib_dir` to point
elsewhere, or `infinity_mode: "remote"` plus `infinity_addr` to use a server.

The first run writes `~/Library/Application Support/rgrep/config.json`
(`~/.config/rgrep` on Linux). `--home <dir>` puts it elsewhere.

## Commands

| command | description |
|---|---|
| `rgrep index [path] [-y] [--force] [--in-process]` | incremental indexing; `--force` re-extracts everything |
| `rgrep search <query> [-k N] [--alpha A] [--path P] [--lang L] [--kind K] [--no-tests] [--code] [--json]` | hybrid search |
| `rgrep get <path>:<line>` | the smallest code unit containing that line |
| `rgrep status / stats / config [--json]` | state, including the Infinity endpoint |
| `rgrep serve` | run the daemon in the foreground |
| `rgrep shutdown` | stop the daemon |
| `rgrep version` | version and native ABI |

Exit codes: `0` success, `2` usage error, `4` no results — so an agent can tell
the cases apart.

## How indexing and search work

**1) Splitting** (`native/rgrep_native`)

tree-sitter extracts `function/method/class/struct/interface/trait/impl/const/raw`
per language by walking the CST (no S-expression parsing), and collects the
signature (up to `body`), the doc comment (previous comment siblings plus Python
docstrings) and the call names (`call_expression` function field). Files over
1 MB are skipped; a file with no declaration falls back to 80-line `raw` units.

**2) Embedded text**

```
File: <path>
Lang: <lang>
Kind: <kind>  Name: <name>  Parent: <parent>
Signature: ...
Doc: ...
Calls: ...
Code:
<code>
```

`File:` comes before `Code:`, so truncation (8000 chars by default) only drops
the tail of the code.

**3) Encoding**

`[CLS] [D|Q] ...tokens... [SEP]`, documents truncated to 512 tokens and queries
to 256, output `[B,T,48]`, then CLS/prefix/SEP are dropped and every token vector
is L2-normalised. Batches of 8.

**4) Storage**

One row per code unit in the Infinity table `rgrep.code_units`:

| column | role |
|---|---|
| `path`, `lang`, `kind`, `name`, `parent`, `start_line`, `end_line` | metadata and filters |
| `text` | the text that was embedded |
| `fts` | identifier-split text (`getUserById` → `get user by id`) for BM25 |
| `code` | the snippet returned to callers |
| `vec` | `MULTIVECTOR(FLOAT,48)` of the ColBERT token vectors |

Indexes: `USING PLAID` on `vec`, full-text on `fts`. Both are created at startup,
before the table has any rows. The rows themselves live in
`$DATA_DIR/infinity` (`catalog/`, `data/`, `wal/`), written by the embedded
engine and readable only through it.

The SQLite catalog (`$DATA_DIR/rgrep.db`) holds only `files` (hash, mtime, size,
lang, unit count per file) and `meta`. It never holds text or vectors.

**5) Retrieval** — one statement, executed inside Infinity:

```
MATCH TENSOR (vec, <query token vectors>, topn=tensor_topn)   -- PLAID + MaxSim
MATCH TEXT   (fts, "<identifier terms>", topn=text_topn)      -- BM25
FUSION       ('weighted_sum', 'weights=<alpha>,<1-alpha>;normalize=minmax')
```

then rgrep re-ranks the fused candidate list:

```
→ boost: exact definition name +0.55 / name prefix +0.2 / function +0.06 / class +0.04
→ penalty: test files ×0.55, examples/docs ×0.85
→ at most 2 units per file → top-k
```

`--alpha` is the semantic weight of the engine-side fusion; with `"fusion":
"rrf"` in the config the ranking is RRF instead and `--alpha` is ignored.
`path`, `lang` and `kind` filters are pushed into the engine. `--no-tests` is
applied in Go, because "is this a test" is rgrep's own heuristic.

## Measurements (M1 / CPU / int8 model)

| case | number |
|---|---|
| rgrep itself (16 files / 241 units), full index | 2.8 s |
| model2vec (53 files / 592 units), full index | 7.7 s (≈13 ms/unit) |
| re-index with no changes | 1 ms (every stat hits) |
| `search` (daemon, including CLI start) | **~50 ms** |
| `search --in-process` (including model load) | ~210 ms |
| quality sample (model2vec) | `quantize embeddings to int8` → `quantization.py:quantize_embeddings` rank 1 |

The 4× gap between the daemon and `--in-process` is the cost of loading the
model every time.

## Configuration (`$DATA_DIR/config.json`)

```jsonc
{
  "data_dir": "...",
  "model_dir": "/path/to/LateOn-Code-edge",  // contains model_int8.onnx + tokenizer.json
  "ort_lib_dir": "bin/onnxruntime/lib",      // libonnxruntime.*

  "infinity_mode": "embedded",               // or "remote"
  "infinity_lib_dir": "",                    // default: <exe dir>/lib
  // Remote mode only. The embedded engine ignores it: it serves a unix socket
  // of its own under data_dir.
  "infinity_addr": "127.0.0.1:23817",
  "infinity_db": "rgrep",
  "infinity_table": "code_units",
  "tensor_topn": 200,                        // rows from the PLAID branch
  "text_topn": 200,                          // rows from the BM25 branch
  "fusion_topn": 200,                        // fused rows rgrep re-ranks
  "fusion": "weighted_sum",                  // or "rrf"

  "dim": 48, "doc_len": 512, "query_len": 256,
  "cls_id": 50281, "sep_id": 50282,
  "prefix_q": 50368, "prefix_d": 50369,
  "batch_size": 8,
  "max_file_bytes": 1048576,
  "max_embed_chars": 8000,
  "idle_timeout": "30m",
  "ignore": []
}
```

`RGREP_INFINITY_ADDR` switches to remote mode against that endpoint (useful for
tests); `RGREP_INFINITY_MODE` overrides the mode. The embedded engine keeps its
segments in `$DATA_DIR/infinity` and serves `$DATA_DIR/infinity.sock`.

`onnxruntime_go` is built against `ORT_API_VERSION 29`, so it **needs ORT
1.29.x**; on a mismatch the process prints `The requested API version ... is not
available`. Without a usable ORT or model, rgrep falls back to a pure-Go hash
encoder: the pipeline still runs, but the vectors carry no meaning.

## Known limits

- **No fallback index**: if the engine cannot start (missing shared library) or
  a remote server is unreachable, `index` and `search` fail instead of silently
  degrading. The catalog is not a fallback search index.
- **The embedded engine is one process**: `rgrep index` and `rgrep serve` each
  open the data dir, so two of them cannot run at once (the second reports the
  RocksDB lock). The daemon is the intended single writer; `--in-process` is for
  one-shot runs while it is stopped.
- **Filter expressions**: pushed-down filters cannot use `LIKE` (the engine's
  expression parser rejects it), so `--path` becomes a byte range
  `path >= p AND path <= p\uffff`. A path containing characters above U+FFFF
  (emoji) would fall outside that range.
- **Per-file re-encoding**: editing one function re-encodes every unit of that
  file (same as colgrep).
- **Single in-flight query**: the daemon serves one request at a time; there is
  no queue and no concurrency limit.
- **Synchronous indexing**: `index` runs inside the request; progress reporting
  would need a job plus polling.
- **No watcher**: indexing is triggered by `index`, run by hand or from a script.
- **Six languages**: go / python / rust / javascript / typescript / tsx.

## Layout

```
cmd/rgrep             entry point
internal/cli          command parsing and output
internal/daemon       unix-socket server + client + lifecycle (instance.lock, idle exit)
internal/app          wires config + catalog + Infinity backend + encoder
internal/index        walk / incremental plan / unit building / batch encoding
internal/search       post-ranking on top of the engine's fused candidates
internal/infinity     the Infinity backend (Go SDK) + the embedded engine's
                      loader (dlopen, start/stop tied to the process)
internal/store        the code-unit model + the SQLite catalog (files/meta)
internal/embed        Embedder interface + ONNX + hash fallback
internal/native       cgo bindings to the splitter (Rust)
native/rgrep_native   Rust: tree-sitter + tokenizers, JSON out
bin/lib               libinfinity_embedded — the engine, loaded at runtime
docs/                 integration notes
```
