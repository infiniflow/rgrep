# Infinity integration

rgrep does not search locally. Every vector query, every full-text query and the
fusion of the two is executed by Infinity, through the public Go SDK
(`github.com/infiniflow/infinity-go-sdk`). This document records what rgrep
sends, what it expects back, and which behaviours were verified against the
engine rather than assumed.

The short version of the split:

| concern | owner | why |
|---|---|---|
| PLAID multi-vector recall (MaxSim) | Infinity | it is a vector index |
| BM25 full-text recall | Infinity | it is an inverted index |
| fusing the two branches | Infinity (`FUSION`) | the engine ranks, rgrep should not re-invent it |
| which files changed since last pass | SQLite catalog | a local filesystem question |
| test-path penalties, name boosts, per-file cap | rgrep | code-specific policy, not retrieval |

SQLite is therefore **not** a fallback search index. If Infinity is unreachable,
`rgrep index` and `rgrep search` fail with the connection error. A silent
fallback would hide a broken deployment and return worse results than an error.

## Embedded mode (default)

A local code search should not need a database server installed, a port picked,
or a supervisor to keep the two processes in step. So the engine runs **inside
the rgrep daemon**:

```
rgrep serve ── dlopen ──▶ libinfinity_embedded
     │                        │
     │  unix socket           ├─ Engine::open($DATA_DIR/infinity)
     └──── $DATA_DIR/infinity.sock ── PLAID/FULLTEXT/FUSION
```

- `make engine` builds `libinfinity_embedded` from the engine checkout and
  copies it to `bin/lib`; the daemon loads it with `dlopen` (`infinity_lib_dir`
  points elsewhere if needed). No link-time dependency, and a missing library is
  a clear error rather than a link failure.
- `inf_embedded_start(data_dir, socket)` opens the engine and serves the
  ordinary thrift front end on `$DATA_DIR/infinity.sock`. It returns once the
  socket exists, so the first client never races the listener.
- `inf_embedded_stop` signals the serving thread, joins it, unlinks the socket
  and closes the engine, releasing the RocksDB lock on the data directory.
  `App.Close` (daemon exit, `shutdown`, a one-shot `--in-process` run) calls it,
  so the engine's lifetime is exactly the daemon's.
- No TCP listener is opened: the daemon holds only `daemon.sock` (its own API)
  and `infinity.sock` (the engine).

Why a socket and not a function-call ABI: everything above the transport —
thrift framing, the session table, every statement arm, the panic boundary —
already exists in the engine's `server` crate, and a client that speaks thrift
over a unix socket is the *same client* as one that speaks it over TCP. The only
SDK change is a dialer (`infinity.UnixAddress`, `NewInfinityConnectionUnix`), so
`Database`/`Table`/`MatchTensor`/`Fusion` behave identically in both modes. A
`select`-level C ABI would have meant a second statement path to keep in step,
and the SQL surface cannot express hybrid search anyway (`MATCH TENSOR`/`MATCH
TEXT` exist only in the thrift `search_exprs`).

`infinity_mode: "remote"` (or `RGREP_INFINITY_ADDR=host:port`) keeps the old
behaviour: connect to a server somebody else runs.

## Interface rule

rgrep uses the SDK's own vocabulary — `infinity.Connect`, `Database`,
`Table`, `MatchTensor`, `MatchText`, `Fusion`, `Filter`, `Output`, `Limit`,
`Sort`, `ToResult` — and nothing rgrep-specific. When a capability is missing it
is added to the SDK, not hidden behind a private ABI. That keeps two things
true: anything rgrep can do, another client of the SDK can do; and an rgrep bug
is reproducible with a few SDK calls. The embedded mode follows the same rule:
it is a transport the SDK speaks, not a second API.

## Schema

Database `rgrep`, table `code_units`, one row per code unit:

```sql
CREATE TABLE code_units (
    path       VARCHAR,
    lang       VARCHAR,
    kind       VARCHAR,
    name       VARCHAR,
    parent     VARCHAR,
    start_line INTEGER,
    end_line   INTEGER,
    text       VARCHAR,   -- the text that was embedded
    fts        VARCHAR,   -- identifier-split text, BM25 side
    code       VARCHAR,   -- snippet returned to callers
    vec        MULTIVECTOR(FLOAT, 48)
);
CREATE INDEX idx_vec ON code_units (vec) USING PLAID;
CREATE INDEX idx_fts ON code_units (fts) USING FULLTEXT;
```

The SDK spells the multi-vector type `"multivector,48,float32"` (comma form, not
the SQL `MULTIVECTOR(FLOAT,48)` form). Insert values are `[][]float32` — one
slice of 48 floats per token vector, which the SDK encodes as a `DoubleTensor`
literal.

Both indexes are created at startup, before any row exists. PLAID training then
happens on insert. This was verified explicitly, because "create index on an
empty table" is the kind of thing that works in one engine and not another.

## Write path

One file at a time, idempotent:

```go
table.Delete("path = 'internal/cli/cli.go'")
table.Insert([]map[string]interface{}{ /* rows, batched */ })
```

Deleting first makes a re-index of the same file a replacement, not a duplicate,
and makes a crash between the two steps harmless (the next pass deletes again).
The catalog row is written **after** the insert, so a crash leaves the file
"changed" and it is re-sent — the other order would remember units that were
never stored.

The catalog (`$DATA_DIR/rgrep.db`) keeps:

```sql
files(path PRIMARY KEY, hash, mtime_ns, size, lang, units, updated_ns)
meta(key PRIMARY KEY, value)
```

`units` is the per-file unit count, so `rgrep status` can report both totals
without querying Infinity.

## Read path

One statement per query, all inside the engine:

```go
table.
    MatchTensor("vec", queryVectors, "float", tensorTopN, nil).   // PLAID + MaxSim
    MatchText("fts", identifierTerms, textTopN, nil).             // BM25
    Fusion(fusionMethod, fusionTopN, fusionParams).
    Filter(pushedDownFilter).
    Output([]string{"path", "lang", "kind", "name", "parent",
                    "start_line", "end_line", "code", "score()"}).
    Limit(k).
    ToResult()
```

Details that matter and were verified:

- **Score projection.** `score()` is the projection that returns the score of
  the row after fusion. The response column is named `SCORE`. `#_score` (the
  internal name) is rejected by the expression parser, and `ROW_ID()` is not
  needed.
- **Fusion parameters.** `weighted_sum` takes `weights=<alpha>,<1-alpha>` and
  optional `normalize=minmax|none`; `rrf` takes `rank_constant`. rgrep maps
  `--alpha` onto the semantic weight, so `--alpha 0.7` becomes
  `weights=0.7000,0.3000`. With `"fusion": "rrf"` in the config, `--alpha` is
  ignored.
- **Single-branch queries.** When the query has no identifier terms, only the
  tensor branch and no `FUSION` are sent; when it has no vectors, only
  `MATCH TEXT`. Both still support `score()`.
- **Default operator.** `MATCH TEXT` uses OR across terms, so a symbol query
  such as `split_args split args` matches a document containing either. rgrep
  relies on that and does not emit `AND`.
- **`LIKE` is not accepted in a pushed-down filter** (the expression parser
  answers "the filter must be a boolean expression"). `--path` is therefore a
  byte range, `path >= 'internal/' AND path <= 'internal/\uffff'`; `lang` and
  `kind` are plain equalities, joined with `AND`.
- **`--no-tests` stays in Go.** "Is this path a test" is rgrep's heuristic
  (`_test.go`, `/tests/`, `.spec.`, …), not something the engine should know.
- **The table object is reusable.** `ToResult` resets the query builder, so one
  `*Table` serves every query; a fresh builder is not needed per call.

`UnitAt` (used by `rgrep get`) is a filtered read, ordered by `start_line`
ascending and limited to one row, which yields the innermost enclosing unit.

## Ranking policy that stayed in rgrep

The engine returns a fused candidate list; rgrep then applies what is specific
to code:

```
name is exactly a query token       +0.55
name starts with a query token      +0.20
function / method                   +0.06
class / struct / interface / trait  +0.04
raw fallback unit                   -0.05
path looks like a test              ×0.55
path looks like examples/docs       ×0.85
at most two units per file
```

This is a multiplication on the engine's fused score, so it can reorder
candidates but cannot invent them.

## Configuration

```jsonc
{
  "infinity_mode": "embedded",   // or "remote"
  "infinity_lib_dir": "",        // default: <exe dir>/lib
  "infinity_addr": "127.0.0.1:23817",   // remote mode only
  "infinity_db": "rgrep",
  "infinity_table": "code_units",
  "tensor_topn": 200,
  "text_topn": 200,
  "fusion_topn": 200,
  "fusion": "weighted_sum"      // or "rrf"
}
```

Derived in embedded mode: `$DATA_DIR/infinity` (segments) and
`$DATA_DIR/infinity.sock` (the engine's socket). `RGREP_INFINITY_ADDR` forces
remote mode against that endpoint, which is how the integration test is pointed
at a running server.

## Testing

The backend test starts an **embedded** engine in a temporary directory, so it
needs no server:

```bash
go test ./internal/infinity
```

It creates a database named `rgrep_test_<unix-nano>`, writes two units, and
checks semantic recall, hybrid recall, prefix filtering, unit lookup and
delete-by-path. It skips only when `bin/lib` has no engine library (`make
engine`); `RGREP_INFINITY_ADDR=host:port` points it at a running server instead,
and `RGREP_INFINITY_LIB_DIR` at another library location.

## Verified engine behaviours

Everything below was executed against a `rust_engine` build, not inferred from
the C++ implementation:

| behaviour | result |
|---|---|
| `MULTIVECTOR` column, `[][]float32` insert | stored and searched correctly |
| PLAID index created before any row | insert afterwards works, recall works |
| PLAID with default parameters (no `nbits`/`n_centroids`) | works; the engine chooses |
| `MATCH TENSOR` without any index | works (exact scan) |
| `MATCH TEXT` + `MATCH TENSOR` + `FUSION('rrf')` | works over Thrift |
| `FUSION('weighted_sum', 'weights=0.6,0.4')` | works; `score()` returns the fused value |
| `score()` in the projection | returned as column `SCORE` |
| `Filter` with `=`, `!=`, `>=`, `<`, `AND`, `OR` | works |
| `Filter` with `LIKE` | rejected by the expression parser |
| `Delete(filter)` | works |
| `Sort` + `Limit` on a filtered read | works |
| thrift over a unix socket (embedded engine) | works: same SDK calls, `UnixAddress` dialer |
| engine inside the host process, no TCP listener | verified with `lsof`: only unix sockets |
| stop removes the socket, data reopens | verified: restart keeps the rows, no stale lock |
| a VARCHAR cell containing a TAB | fixed in the engine; the stored row text escapes it |
