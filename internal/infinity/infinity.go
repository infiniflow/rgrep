// Package infinity is rgrep's retrieval backend.
//
// Everything that is a search — the PLAID multi-vector recall and the BM25
// full-text recall, and the fusion of the two — runs inside Infinity, through
// the public Go SDK. rgrep sends one SELECT per query:
//
//	MATCH TENSOR (vec, <query token vectors>, 'float', 'maxsim', <n>)
//	MATCH TEXT   (fts, '<query terms>', <n>)
//	FUSION       ('weighted_sum' | 'rrf', <n>[, 'weights=…'])
//
// and only re-ranks what comes back (name boosts, test-path penalties, at most
// two hits per file). It never scores vectors itself.
//
// The document table is deliberately plain: one row per code unit, a
// MULTIVECTOR column for the ColBERT token vectors, a VARCHAR column for the
// identifier-split text. Both get an index at open time, so the SQL above is
// served by PLAID and by the inverted index respectively.
//
// The filter is answered from secondary indexes, not by re-reading rows: `lang`
// and `kind` are low-cardinality secondary indexes, so the engine turns
// `lang = 'go' AND kind = 'function'` into two bitmap reads and intersects them
// while building the candidate mask of the recall. `path` carries one too, which
// answers `path = ?` (the delete of a file being re-indexed, and `rgrep get`).
// A path *prefix* cannot use it - a VARCHAR secondary key is a hash, so it has
// no order to scan - and stays a residual predicate.
package infinity

import (
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/apache/thrift/lib/go/thrift"
	sdk "github.com/infiniflow/infinity-go-sdk"

	"github.com/infiniflow/rgrep/internal/store"
)

// Document table columns.
const (
	ColPath    = "path"
	ColLang    = "lang"
	ColKind    = "kind"
	ColName    = "name"
	ColParent  = "parent"
	ColStart   = "start_line"
	ColEnd     = "end_line"
	ColText    = "text"
	ColFTS     = "fts"
	ColCode    = "code"
	ColVectors = "vec"
)

// scoreColumn is what Infinity names a `score()` projection in the response.
const scoreColumn = "SCORE"

// Fusion methods the backend knows how to ask for.
const (
	FusionWeightedSum = "weighted_sum"
	FusionRRF         = "rrf"
)

// Vector column shapes. The two are not interchangeable: a table is created
// for one of them, and switching means a new table (the column type is part of
// the schema, not a query parameter).
const (
	// VectorMulti stores one vector per token in a MULTIVECTOR column and is
	// recalled by PLAID + MaxSim.
	VectorMulti = "multivector"
	// VectorDense stores one pooled vector per row in a VECTOR column and is
	// recalled by HNSW under Metric.
	VectorDense = "vector"
)

// Config is the connection and query shape of one Infinity instance.
type Config struct {
	Host     string
	Port     int
	Database string
	Table    string

	// Embedded runs the engine inside this process (see `Embedded`): `DataDir`
	// is where it keeps its segments, `Socket` the unix socket it serves and
	// `LibDir` the directory holding the shared library. When Embedded is
	// false, Host/Port name a server somebody else runs.
	Embedded bool
	DataDir  string
	Socket   string
	LibDir   string

	// Dim is the encoder width: the MULTIVECTOR column is (Dim), the VECTOR
	// column is (Dim) as well.
	Dim int
	// VectorType is VectorMulti (default) or VectorDense, and decides both the
	// column type and the recall operator.
	VectorType string
	// Metric is the HNSW distance for VectorDense: "cosine" (default), "l2"
	// or "ip". It is ignored by the PLAID index, which is always MaxSim.
	Metric string
	// HNSWM / HNSWEfConstruction are the HNSW build parameters. Zero leaves the
	// engine's defaults (M=16, ef_construction=200).
	HNSWM              int
	HNSWEfConstruction int
	// TensorTopN / TextTopN are how many rows each branch contributes, and
	// FusionTopN how many fused rows the engine keeps before rgrep re-ranks.
	TensorTopN int
	TextTopN   int
	FusionTopN int
	// Fusion is FusionWeightedSum (default) or FusionRRF.
	Fusion string
	// BatchSize is how many units go into one INSERT.
	BatchSize int
	// PlaidCentroids is the coarse-centroid budget of the PLAID index: 0 lets
	// the engine size it from the token count, which on a large repo asks for
	// tens of thousands of centroids and turns one index pass into hours of
	// k-means. See `indexSpecs`.
	PlaidCentroids int
	// VecIndex is the multi-vector recall index: "plaid" (default) or "smve".
	// SMVEWidth / SMVETopK are its projection width and per-side token budget.
	VecIndex  string
	SMVEWidth int
	SMVETopK  int
	// Timeout bounds both connect and each RPC.
	Timeout time.Duration
}

func (c *Config) withDefaults() {
	if c.Host == "" {
		c.Host = "127.0.0.1"
	}
	if c.Port == 0 {
		c.Port = 23817
	}
	if c.Database == "" {
		c.Database = "rgrep"
	}
	if c.Table == "" {
		c.Table = "code_units"
	}
	if c.Dim <= 0 {
		c.Dim = 48
	}
	if c.VectorType == "" {
		c.VectorType = VectorMulti
	}
	if c.Metric == "" {
		c.Metric = "cosine"
	}
	if c.TensorTopN <= 0 {
		c.TensorTopN = 200
	}
	if c.TextTopN <= 0 {
		c.TextTopN = 200
	}
	if c.FusionTopN <= 0 {
		c.FusionTopN = 200
	}
	if c.Fusion == "" {
		c.Fusion = FusionWeightedSum
	}
	if c.BatchSize <= 0 {
		c.BatchSize = 64
	}
	if c.Timeout <= 0 {
		// Not a query deadline: a write's own RPC can carry the engine's
		// checkpoint. The engine seals its append buffer at 512 MB, which is
		// about 9 k units, and the checkpoint that follows writes the block
		// mirror before it answers - seconds of work inside the insert that
		// crossed the line. At the old 10 s the first such insert came back as
		// "i/o timeout" and aborted a full index run at exactly that size
		// (measured, twice, at 9336 units).
		c.Timeout = 120 * time.Second
	}
}

// Addr is "host:port".
func (c *Config) Addr() string { return fmt.Sprintf("%s:%d", c.Host, c.Port) }

// Query is one hybrid retrieval request.
type Query struct {
	// Vectors is the dense-vector path: qn*Dim L2-normalised query token
	// vectors, one per token (VectorMulti).
	Vectors [][]float32
	// Vector is the single pooled query vector (VectorDense). Set exactly one
	// of Vectors and Vector, matching the table's column type.
	Vector []float32
	// Text is the full-text side; empty means keyword recall is skipped.
	Text string
	// K is how many fused rows rgrep wants before its own re-ranking.
	K     int
	Alpha float64 // weight of the semantic branch for weighted_sum
	// Options are the recall operator's own knobs (`plaid_n_ivf_probe`,
	// `plaid_n_doc_to_score`, `plaid_n_full_scores`, ...), passed through to
	// MATCH TENSOR as `k=v;k=v`. Nil keeps the engine's defaults.
	Options map[string]string
	// PathPrefix, Lang and Kind are pushed down as a filter.
	PathPrefix string
	Lang       string
	Kind       string
}

// Hit is one row returned by the engine.
type Hit struct {
	Path      string
	Lang      string
	Kind      string
	Name      string
	Parent    string
	StartLine int
	EndLine   int
	Score     float64
	Code      string
}

// Backend is a connected Infinity instance holding the code-unit table.
type Backend struct {
	cfg      Config
	conn     *sdk.InfinityConnection
	db       *sdk.Database
	tbl      *sdk.Table
	embedded *Embedded
	// mu serialises engine calls. The SDK client is one thrift connection with
	// one request/response stream, so two goroutines sharing it interleave their
	// frames; the daemon serves one goroutine per client, which is how a query
	// issued while an index pass was running came back as a socket timeout and
	// then poisoned the connection for every later call.
	mu sync.Mutex
}

// Open connects, then makes sure the database, the table and both indexes
// exist. Creating an index on an empty table is supported, so the table can be
// prepared before the first `rgrep index` pass runs.
//
// In embedded mode it also *starts* the engine, which then lives exactly as
// long as this Backend: nobody has to run `infinity serve`, and nothing
// survives the process.
func Open(cfg Config) (*Backend, error) {
	cfg.withDefaults()
	conf := &thrift.TConfiguration{ConnectTimeout: cfg.Timeout, SocketTimeout: cfg.Timeout}
	var (
		conn     *sdk.InfinityConnection
		embedded *Embedded
	)
	if cfg.Embedded {
		e, err := StartEmbedded(cfg.LibDir, cfg.DataDir, cfg.Socket)
		if err != nil {
			return nil, err
		}
		embedded = e
		conn, err = sdk.NewInfinityConnectionUnix(cfg.Socket, conf)
		if err != nil {
			_ = embedded.Stop()
			return nil, fmt.Errorf("infinity (embedded) %s is not reachable: %w", cfg.Socket, err)
		}
	} else {
		var err error
		conn, err = sdk.NewInfinityConnectionWithConfig(
			sdk.NetworkAddress{IP: cfg.Host, Port: cfg.Port}, conf)
		if err != nil {
			return nil, fmt.Errorf("infinity %s is not reachable: %w", cfg.Addr(), err)
		}
	}

	b := &Backend{cfg: cfg, conn: conn, embedded: embedded}
	if err := b.ensureSchema(); err != nil {
		_, _ = conn.Disconnect()
		_ = b.closeEngine()
		return nil, err
	}
	return b, nil
}

// reconnect drops the client and dials a new one. The engine's thrift client is
// one connection with one request/response stream: an RPC that hit the socket
// timeout leaves its reply queued, so the next call reads the previous call's
// frame (`Select: out of order sequence response`) or blocks again, and the
// daemon then fails every query until it is restarted. Rebuilding the client is
// the only way back.
func (b *Backend) reconnect() error {
	if b.conn != nil {
		_, _ = b.conn.Disconnect()
		b.conn = nil
	}
	conf := &thrift.TConfiguration{ConnectTimeout: b.cfg.Timeout, SocketTimeout: b.cfg.Timeout}
	var (
		conn *sdk.InfinityConnection
		err  error
	)
	if b.cfg.Embedded {
		conn, err = sdk.NewInfinityConnectionUnix(b.cfg.Socket, conf)
	} else {
		conn, err = sdk.NewInfinityConnectionWithConfig(
			sdk.NetworkAddress{IP: b.cfg.Host, Port: b.cfg.Port}, conf)
	}
	if err != nil {
		return fmt.Errorf("reconnect %s: %w", b.Addr(), err)
	}
	db, err := conn.GetDatabase(b.cfg.Database)
	if err != nil {
		_, _ = conn.Disconnect()
		return fmt.Errorf("reconnect %s: %w", b.Addr(), err)
	}
	tbl, err := db.GetTable(b.cfg.Table)
	if err != nil {
		_, _ = conn.Disconnect()
		return fmt.Errorf("reconnect %s: %w", b.Addr(), err)
	}
	b.conn, b.db, b.tbl = conn, db, tbl
	return nil
}

// read runs one engine read, and retries it once on a fresh connection if the
// first attempt failed. Reads are idempotent, so the retry cannot duplicate
// anything; writes are not wrapped, and rely on the indexer's own recovery
// (`ReplaceUnits` deletes the file's rows before it writes them).
func (b *Backend) read(fn func() error) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	err := fn()
	if err == nil {
		return nil
	}
	if rerr := b.reconnect(); rerr != nil {
		return err
	}
	return fn()
}

// Addr reports the endpoint, for `status`.
func (b *Backend) Addr() string {
	if b.embedded != nil {
		return "embedded:" + b.embedded.Socket()
	}
	return b.cfg.Addr()
}

// closeEngine stops an embedded engine, if this backend started one.
func (b *Backend) closeEngine() error {
	if b.embedded == nil {
		return nil
	}
	e := b.embedded
	b.embedded = nil
	return e.Stop()
}

// Table reports the fully qualified table name, for `status`.
func (b *Backend) Table() string { return b.cfg.Database + "." + b.cfg.Table }

func (b *Backend) ensureSchema() error {
	db, err := b.conn.GetDatabase(b.cfg.Database)
	if err != nil {
		if db, err = b.conn.CreateDatabase(b.cfg.Database, sdk.ConflictTypeIgnore, ""); err != nil {
			return fmt.Errorf("create database %s: %w", b.cfg.Database, err)
		}
	}
	b.db = db

	tbl, err := db.GetTable(b.cfg.Table)
	if err != nil {
		schema := sdk.TableSchema{
			{Name: ColPath, DataType: "VARCHAR"},
			{Name: ColLang, DataType: "VARCHAR"},
			{Name: ColKind, DataType: "VARCHAR"},
			{Name: ColName, DataType: "VARCHAR"},
			{Name: ColParent, DataType: "VARCHAR"},
			{Name: ColStart, DataType: "INTEGER"},
			{Name: ColEnd, DataType: "INTEGER"},
			{Name: ColText, DataType: "VARCHAR"},
			{Name: ColFTS, DataType: "VARCHAR"},
			{Name: ColCode, DataType: "VARCHAR"},
			{Name: ColVectors, DataType: b.vectorColumnType()},
		}
		if tbl, err = db.CreateTable(b.cfg.Table, schema, sdk.ConflictTypeIgnore); err != nil {
			return fmt.Errorf("create table %s: %w", b.cfg.Table, err)
		}
	}
	b.tbl = tbl

	for _, ix := range indexSpecs(b.cfg) {
		// Ask first. `CREATE INDEX ... IF NOT EXISTS` is supposed to be the
		// one call that does that, but this engine's `IF NOT EXISTS` still
		// rewrites the catalog record and rebuilds the index over every row
		// (measured: 260 s per open on 4.6 k units), which made every command
		// after the first one time out. A `SHOW INDEX` is a catalog read.
		if _, err := tbl.ShowIndex(ix.name); err == nil {
			continue
		}
		if _, err := tbl.CreateIndex(ix.name, ix.info, sdk.ConflictTypeIgnore, ""); err != nil {
			return fmt.Errorf("create %s index on %s: %w", ix.what, ix.info.TargetName, err)
		}
	}
	return nil
}

// indexSpec is one index the schema needs.
type indexSpec struct {
	name string
	what string
	info *sdk.IndexInfo
}

// indexSpecs is everything the unit table needs beyond its columns.
//
// The three secondary indexes are the filter's own path through the engine:
//
//   - `idx_path` answers `path = ?`. `ReplaceUnits` deletes a file's rows before
//     it writes the new ones, and that delete without an index is a scan of the
//     whole table - the cost that made a full index run quadratic (313 files in
//     807 s, ending when one delete passed the RPC timeout). The same index makes
//     `UnitAt` (`rgrep get`) a probe plus a line comparison instead of a scan.
//   - `idx_lang` and `idx_kind` answer the search filter. Both hold a handful of
//     distinct values, so they are `cardinality = 'low'`: one roaring bitmap per
//     value, which the engine intersects while building the recall's candidate
//     mask, instead of decoding those columns of every row.
func indexSpecs(cfg Config) []indexSpec {
	secondary := func(column string, low bool) *sdk.IndexInfo {
		var params map[string]string
		if low {
			params = map[string]string{"cardinality": "low"}
		}
		return sdk.NewIndexInfo(column, sdk.IndexTypeSecondary, params)
	}
	var vec indexSpec
	if cfg.dense() {
		// HNSW keeps the graph and the vectors; the parameters that matter for
		// a code index of tens of thousands of units are M (edges per node) and
		// ef_construction (candidate list while inserting). The engine's
		// defaults are 16/200; a smaller graph is cheaper to build and costs
		// recall, so rgrep only overrides them when configured.
		metric := cfg.Metric
		if metric == "" {
			metric = "cosine"
		}
		// The engine upper-cases an option name before checking it against the
		// index type's whitelist (METRIC / M / EF_CONSTRUCTION), so the key is
		// spelled the way a `CREATE INDEX ... WITH (...)` statement spells it.
		params := map[string]string{"metric": metric}
		if cfg.HNSWM > 0 {
			params["m"] = strconv.Itoa(cfg.HNSWM)
		}
		if cfg.HNSWEfConstruction > 0 {
			params["ef_construction"] = strconv.Itoa(cfg.HNSWEfConstruction)
		}
		vec = indexSpec{"idx_" + ColVectors, "HNSW", sdk.NewIndexInfo(ColVectors, sdk.IndexTypeHnsw, params)}
	} else if strings.EqualFold(cfg.VecIndex, "smve") {
		// SMVE turns each token into a sparse top-k code over a random unit
		// projection, sums the query's codes, averages the document's, and
		// searches them with the BMP block-max index. Its candidates are
		// re-ranked with the exact MaxSim read back from the column, so the
		// score it reports is the float32 one - what makes it the reference to
		// check a lossy index against. `width` is the sparse term space, `topk`
		// how many terms each side keeps.
		params := map[string]string{}
		if cfg.SMVEWidth > 0 {
			params["width"] = strconv.Itoa(cfg.SMVEWidth)
		}
		if cfg.SMVETopK > 0 {
			params["topk"] = strconv.Itoa(cfg.SMVETopK)
		}
		vec = indexSpec{"idx_" + ColVectors, "SMVE", sdk.NewIndexInfo(ColVectors, sdk.IndexTypeSMVE, params)}
	} else {
		// `n_centroids` is the one PLAID parameter that decides how long an
		// index pass takes: the engine's own choice is `16*sqrt(tokens)`
		// rounded down to a power of two, so a repo of 130 k units (~16 M token
		// vectors) asks for 32 768 centroids, and `kmeans_l2` over
		// `min(tokens, 256*k)` samples costs `samples * k * dims` per iteration.
		// That is hours, spent before a single query can be answered. A fixed
		// budget keeps the build in seconds; 0 leaves the engine's heuristic.
		var plaid map[string]string
		if cfg.PlaidCentroids > 0 {
			plaid = map[string]string{"n_centroids": strconv.Itoa(cfg.PlaidCentroids)}
		}
		vec = indexSpec{"idx_" + ColVectors, "PLAID", sdk.NewIndexInfo(ColVectors, sdk.IndexTypePLAID, plaid)}
	}
	specs := []indexSpec{
		{"idx_" + ColFTS, "full-text", sdk.NewIndexInfo(ColFTS, sdk.IndexTypeFullText, nil)},
		{"idx_" + ColPath, "path", secondary(ColPath, false)},
		{"idx_" + ColLang, "lang", secondary(ColLang, true)},
		{"idx_" + ColKind, "kind", secondary(ColKind, true)},
	}
	// `vec_index: "none"` leaves the column unindexed on purpose: MATCH TENSOR
	// then scores every row from its float32 tokens, which is the exact MaxSim
	// the approximation is measured against.
	if !strings.EqualFold(cfg.VecIndex, "none") {
		specs = append([]indexSpec{vec}, specs...)
	}
	return specs
}

// dense reports whether the table stores one pooled vector per row.
func (c *Config) dense() bool {
	return c.VectorType == VectorDense
}

// vectorColumnType is the SDK's comma form of the vector column definition:
// `vector,<dim>,float32` for dense, `multivector,<dim>,float32` for ColBERT.
func (b *Backend) vectorColumnType() string {
	if b.cfg.dense() {
		return fmt.Sprintf("vector,%d,float32", b.cfg.Dim)
	}
	return fmt.Sprintf("multivector,%d,float32", b.cfg.Dim)
}

// ReplaceUnits makes the engine's rows for one file exactly `units`: it
// deletes the file's rows and inserts the new ones. An empty `units` therefore
// just drops the file.
func (b *Backend) ReplaceUnits(path string, units []store.Unit) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.deletePath(path); err != nil {
		return err
	}
	if len(units) == 0 {
		return nil
	}
	for start := 0; start < len(units); start += b.cfg.BatchSize {
		end := start + b.cfg.BatchSize
		if end > len(units) {
			end = len(units)
		}
		rows := make([]map[string]interface{}, 0, end-start)
		for i := start; i < end; i++ {
			u := &units[i]
			rows = append(rows, map[string]interface{}{
				ColPath:    u.Path,
				ColLang:    u.Lang,
				ColKind:    u.Kind,
				ColName:    u.Name,
				ColParent:  u.Parent,
				ColStart:   u.StartLine,
				ColEnd:     u.EndLine,
				ColText:    u.Text,
				ColFTS:     u.FTS,
				ColCode:    u.Code,
				ColVectors: b.vectorValue(u.Vec),
			})
		}
		if _, err := b.tbl.Insert(rows); err != nil {
			return fmt.Errorf("insert %s units [%d:%d]: %w", path, start, end, err)
		}
	}
	return nil
}

// DeletePath removes every row of one file.
func (b *Backend) DeletePath(path string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.deletePath(path)
}

// deletePath is the lock-free form, for callers that already hold `mu`.
func (b *Backend) deletePath(path string) error {
	if _, err := b.tbl.Delete(fmt.Sprintf("%s = %s", ColPath, quote(path))); err != nil {
		return fmt.Errorf("delete %s: %w", path, err)
	}
	return nil
}

// Search runs the hybrid query: vector recall (PLAID + MaxSim or HNSW + the
// configured metric), BM25 recall, and the fusion of the two, all inside
// Infinity.
func (b *Backend) Search(q Query) ([]Hit, error) {
	hasText := strings.TrimSpace(q.Text) != ""
	hasDense := len(q.Vector) > 0
	hasMulti := len(q.Vectors) > 0
	if !hasText && !hasDense && !hasMulti {
		return nil, nil
	}
	limit := q.K
	if limit <= 0 {
		limit = b.cfg.FusionTopN
	}
	if limit > b.cfg.FusionTopN {
		limit = b.cfg.FusionTopN
	}

	var (
		res interface{}
		err error
	)
	run := func() error {
		// The recall operator follows the query's shape, which in turn follows
		// the table's column type: MatchTensor over a MULTIVECTOR column and
		// MatchDense over a VECTOR column are not interchangeable.
		var tbl *sdk.Table
		switch {
		case hasMulti && hasText:
			tbl = b.tbl.
				MatchTensor(ColVectors, q.Vectors, "float", b.cfg.TensorTopN, q.Options).
				MatchText(ColFTS, q.Text, b.cfg.TextTopN, nil).
				Fusion(b.fusion(q), limit, b.fusionParams(q))
		case hasDense && hasText:
			tbl = b.tbl.
				MatchDense(ColVectors, q.Vector, "float", b.cfg.Metric, b.cfg.TensorTopN, nil).
				MatchText(ColFTS, q.Text, b.cfg.TextTopN, nil).
				Fusion(b.fusion(q), limit, b.fusionParams(q))
		case hasMulti:
			tbl = b.tbl.MatchTensor(ColVectors, q.Vectors, "float", b.cfg.TensorTopN, q.Options)
		case hasDense:
			tbl = b.tbl.MatchDense(ColVectors, q.Vector, "float", b.cfg.Metric, b.cfg.TensorTopN, nil)
		default:
			tbl = b.tbl.MatchText(ColFTS, q.Text, b.cfg.TextTopN, nil)
		}
		if filter := q.filter(); filter != "" {
			tbl = tbl.Filter(filter)
		}
		res, err = tbl.
			Output([]string{ColPath, ColLang, ColKind, ColName, ColParent, ColStart, ColEnd, ColCode, "score()"}).
			Limit(limit).
			ToResult()
		return err
	}
	if err = b.read(run); err != nil {
		return nil, err
	}
	return hits(res)
}

// UnitAt returns the innermost unit of `path` that encloses `line`.
func (b *Backend) UnitAt(path string, line int) (*store.Unit, error) {
	filter := fmt.Sprintf("%s = %s AND %s <= %d AND %s >= %d",
		ColPath, quote(path), ColStart, line, ColEnd, line)
	var (
		res interface{}
		err error
	)
	if err = b.read(func() error {
		res, err = b.tbl.
			Filter(filter).
			Output([]string{ColPath, ColLang, ColKind, ColName, ColParent, ColStart, ColEnd, ColText, ColCode}).
			Sort([][2]interface{}{{ColStart, sdk.SortTypeAsc}}).
			Limit(1).
			ToResult()
		return err
	}); err != nil {
		return nil, err
	}
	rows, ok := res.(*sdk.QueryResult)
	if !ok {
		return nil, fmt.Errorf("unexpected result type %T", res)
	}
	if n := len(rows.Data[ColPath]); n == 0 {
		return nil, fmt.Errorf("%s:%d is not inside any indexed unit", path, line)
	}
	return &store.Unit{
		Path:      str(rows.Data[ColPath][0]),
		Lang:      str(rows.Data[ColLang][0]),
		Kind:      str(rows.Data[ColKind][0]),
		Name:      str(rows.Data[ColName][0]),
		Parent:    str(rows.Data[ColParent][0]),
		StartLine: num(rows.Data[ColStart][0]),
		EndLine:   num(rows.Data[ColEnd][0]),
		Text:      str(rows.Data[ColText][0]),
		Code:      str(rows.Data[ColCode][0]),
	}, nil
}

// Close disconnects.
func (b *Backend) Close() error {
	var err error
	if b.conn != nil {
		_, err = b.conn.Disconnect()
		b.conn = nil
	}
	// The engine this backend started goes down with it: that is the whole
	// point of the embedded mode, and it also releases the data dir.
	if stopErr := b.closeEngine(); err == nil {
		err = stopErr
	}
	return err
}

func (b *Backend) fusion(q Query) string {
	if b.cfg.Fusion == FusionRRF {
		return FusionRRF
	}
	return FusionWeightedSum
}

func (b *Backend) fusionParams(q Query) map[string]interface{} {
	if b.fusion(q) != FusionWeightedSum {
		return nil
	}
	alpha := q.Alpha
	if alpha <= 0 {
		alpha = 0.6
	}
	if alpha > 1 {
		alpha = 1
	}
	return map[string]interface{}{
		"weights":   fmt.Sprintf("%.4f,%.4f", alpha, 1-alpha),
		"normalize": "minmax",
	}
}

// filter renders the pushed-down predicates. `no_tests` is not here: deciding
// which path is a test needs rgrep's heuristics, so it stays client-side.
//
// The equalities are the ones the engine can answer from `idx_lang`/`idx_kind`:
// each is a bitmap read, and a conjunction of them is intersected before the
// recall starts, so a selective `--lang` no longer costs a decode of that column
// of every row. The prefix is a residual predicate - a VARCHAR secondary key is
// a hash (`ConvertToHashU64`), so no index can order it and a range over it is
// evaluated row by row.
func (q Query) filter() string {
	var parts []string
	if q.PathPrefix != "" {
		lo := quote(q.PathPrefix)
		// A byte-range filter instead of LIKE: the engine's expression parser
		// rejects LIKE in a pushed-down filter (verified against rust_engine).
		parts = append(parts, fmt.Sprintf("%s >= %s AND %s <= %s", ColPath, lo, ColPath, quote(q.PathPrefix+"\uffff")))
	}
	if q.Lang != "" {
		parts = append(parts, fmt.Sprintf("%s = %s", ColLang, quote(q.Lang)))
	}
	if q.Kind != "" {
		parts = append(parts, fmt.Sprintf("%s = %s", ColKind, quote(q.Kind)))
	}
	return strings.Join(parts, " AND ")
}

func hits(res interface{}) ([]Hit, error) {
	rows, ok := res.(*sdk.QueryResult)
	if !ok {
		return nil, fmt.Errorf("unexpected result type %T", res)
	}
	n := len(rows.Data[ColPath])
	out := make([]Hit, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, Hit{
			Path:      str(rows.Data[ColPath][i]),
			Lang:      str(rows.Data[ColLang][i]),
			Kind:      str(rows.Data[ColKind][i]),
			Name:      str(rows.Data[ColName][i]),
			Parent:    str(rows.Data[ColParent][i]),
			StartLine: num(rows.Data[ColStart][i]),
			EndLine:   num(rows.Data[ColEnd][i]),
			Code:      str(rows.Data[ColCode][i]),
			Score:     numf(rows.Data[scoreColumn][i]),
		})
	}
	return out, nil
}

// vectorValue is the column value the SDK encodes for an insert: a plain
// float buffer for a VECTOR column, one `dim`-sized row per token for a
// MULTIVECTOR column.
func (b *Backend) vectorValue(vec []float32) interface{} {
	if b.cfg.dense() {
		if len(vec) > b.cfg.Dim {
			return vec[:b.cfg.Dim]
		}
		return vec
	}
	return reshape(vec, b.cfg.Dim)
}

// reshape turns the flat token buffer into the MULTIVECTOR literal the SDK
// encodes for an insert: one row of `dim`-sized vectors.
func reshape(vec []float32, dim int) [][]float32 {
	if dim <= 0 || len(vec) == 0 {
		return [][]float32{}
	}
	n := len(vec) / dim
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		out[i] = vec[i*dim : (i+1)*dim]
	}
	return out
}

// quote renders a SQL string literal. Paths are the only untrusted input, and
// a single quote is the only character that can end the literal early.
func quote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}

func str(v interface{}) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func num(v interface{}) int {
	switch n := v.(type) {
	case int:
		return n
	case int32:
		return int(n)
	case int64:
		return int(n)
	case float64:
		return int(n)
	case float32:
		return int(n)
	}
	return 0
}

func numf(v interface{}) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case float32:
		return float64(n)
	case int64:
		return float64(n)
	case int:
		return float64(n)
	case string:
		f, _ := strconv.ParseFloat(n, 64)
		return f
	}
	return 0
}
