# rgrep_native

The Rust half of rgrep: tree-sitter code splitting and HuggingFace tokenization,
exposed as a C ABI that Go reaches through cgo.

Nothing else lives here. Walking files, hashing, choosing what changed, building
the embedded text and talking to Infinity are Go's job.

## Build

```bash
# From the repo root, which is what `make build` does for you:
cd native/rgrep_native && cargo build --release
```

That produces `target/release/librgrep_native.a`, which `internal/native/native.go`
links via `#cgo LDFLAGS`. `make native` runs the same command; `make native-test`
runs the crate's tests.

The crate builds offline if the dependency set is already in the cargo cache
(`tree-sitter` 0.25 with the go / python / rust / javascript / typescript
grammars, `tokenizers` 0.21 built against `fancy-regex` instead of onig, which
keeps the link free of a native regex library).

## ABI

`include/rgrep_native.h` is the contract; `internal/native` is the only caller.

| symbol | returns |
|---|---|
| `rg_codesplit(lang, path, content, len, err)` | JSON: `language`, `degraded`, `truncated`, `units[]` |
| `rg_tokenize(tokenizer_path, text, len, err)` | JSON: `{"ids": [...]}`, no special tokens |
| `rg_languages()` | JSON array of supported language ids |
| `rg_abi_version()` | schema version, matching `native.ABIExpected` in Go |
| `rg_free(ptr)` | releases any string this library returned |

A call that fails returns `NULL` and, when `err` is non-null, stores a message
there. Both are released with `rg_free`.

## What a unit is

One declaration: `function`, `method`, `class`, `struct`, `interface`, `trait`,
`impl`, `enum`, `type`, `const`, `var`, `static`, or `raw` (the Go side falls
back to fixed windows when a file yields nothing).

Each unit carries:

- `name`, and `parent` — the enclosing class/impl/type. A Go method has no
  enclosing node, so its parent comes from the receiver (`func (s *Server) Start`
  → `Server`).
- `signature` — the declaration text up to its body, whitespace collapsed.
- `doc` — the comment block directly above it (a trailing comment on the same
  line is not a doc), plus the Python docstring for a function or class.
- `calls` — the call targets inside the body, in source order, deduplicated.
- `start_line` / `end_line` (1-based) and `start_byte` / `end_byte`, which is how
  Go slices the code back out of the file it already has.

## Tests

```bash
cargo test          # splitting per language, the ABI surface, the tokenizer path
```

The tokenizer test needs a real `tokenizer.json` and skips without
`RGREP_TOKENIZER=/path/to/tokenizer.json`; the rest run on the grammars alone.
