package infinity

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	sdk "github.com/infiniflow/infinity-go-sdk"

	"github.com/infiniflow/rgrep/internal/store"
)

// The backend test needs an engine. By default it starts an embedded one in a
// temporary directory (which is also what rgrep does in production, so the test
// exercises the same path); `RGREP_INFINITY_ADDR=host:port` points it at a
// running server instead:
//
//	RGREP_INFINITY_ADDR=127.0.0.1:23817 go test ./internal/infinity
func openTestBackend(t *testing.T) *Backend {
	t.Helper()
	dbName := fmt.Sprintf("rgrep_test_%d", time.Now().UnixNano())
	cfg := Config{
		Database: dbName, Table: "code_units",
		Dim: 4, BatchSize: 2, Timeout: 5 * time.Second,
	}
	if addr := os.Getenv("RGREP_INFINITY_ADDR"); addr != "" {
		cfg.Host, cfg.Port = splitHostPort(addr)
	} else {
		dir := t.TempDir()
		cfg.Embedded = true
		cfg.DataDir = dir + "/infinity"
		cfg.Socket = dir + "/infinity.sock"
		cfg.LibDir = testLibDir()
		if _, err := findLib(cfg.LibDir); err != nil {
			t.Skipf("no embedded engine: %v (set RGREP_INFINITY_ADDR to use a server)", err)
		}
	}
	b, err := Open(cfg)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = b.Close() })
	return b
}

// testLibDir is where `make engine` puts the shared library, relative to this
// package: `<repo>/bin/lib`.
func testLibDir() string {
	if dir := os.Getenv("RGREP_INFINITY_LIB_DIR"); dir != "" {
		return dir
	}
	abs, err := filepath.Abs("../../bin/lib")
	if err != nil {
		return ""
	}
	return abs
}

func TestReplaceSearchAndDelete(t *testing.T) {
	b := openTestBackend(t)

	units := []store.Unit{
		{Path: "pkg/quantize.go", Lang: "go", Kind: "function", Name: "quantizeEmbeddings",
			StartLine: 10, EndLine: 20, Text: "quantize embeddings to int8",
			FTS: "quantize embeddings int8 quantization", Code: "func quantizeEmbeddings() {}",
			Dim: 4, Vec: []float32{1, 0, 0, 0, 0, 1, 0, 0}},
		{Path: "pkg/parse.go", Lang: "go", Kind: "function", Name: "parseArguments",
			StartLine: 1, EndLine: 8, Text: "parse command line arguments",
			FTS: "parse arguments command line", Code: "func parseArguments() {}",
			Dim: 4, Vec: []float32{0, 0, 1, 0}},
	}
	if err := b.ReplaceUnits("pkg/quantize.go", units[:1]); err != nil {
		t.Fatalf("replace: %v", err)
	}
	if err := b.ReplaceUnits("pkg/parse.go", units[1:]); err != nil {
		t.Fatalf("replace: %v", err)
	}

	// Semantic recall: the first unit's vector is the query.
	hits, err := b.Search(Query{Vectors: [][]float32{{1, 0, 0, 0}}, K: 5, Alpha: 0.6})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(hits) == 0 || hits[0].Path != "pkg/quantize.go" {
		t.Fatalf("semantic hits = %+v", hits)
	}
	if hits[0].Code == "" {
		t.Fatal("code column is empty")
	}

	// Keyword recall plus fusion.
	hits, err = b.Search(Query{
		Vectors: [][]float32{{1, 0, 0, 0}},
		Text:    "parse arguments",
		K:       5,
		Alpha:   0.6,
	})
	if err != nil {
		t.Fatalf("hybrid search: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("hybrid hits = %+v", hits)
	}
	if hits[0].Path != "pkg/quantize.go" {
		t.Fatalf("hybrid order = %+v", hits)
	}
	// Scores are fused by the engine; rgrep only needs them to be ordered.
	for i := 1; i < len(hits); i++ {
		if hits[i].Score > hits[i-1].Score {
			t.Fatalf("fused scores out of order: %+v", hits)
		}
	}

	// Filters are pushed down.
	hits, err = b.Search(Query{Vectors: [][]float32{{1, 0, 0, 0}}, K: 5, PathPrefix: "pkg/parse"})
	if err != nil {
		t.Fatalf("filtered search: %v", err)
	}
	if len(hits) != 1 || hits[0].Path != "pkg/parse.go" {
		t.Fatalf("prefix filter hits = %+v", hits)
	}

	// The equality filters are answered from the secondary indexes, which the
	// engine intersects while it builds the recall's candidate mask.
	hits, err = b.Search(Query{Vectors: [][]float32{{1, 0, 0, 0}}, K: 5, Lang: "go", Kind: "function"})
	if err != nil {
		t.Fatalf("indexed filter search: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("indexed filter hits = %+v", hits)
	}
	hits, err = b.Search(Query{Vectors: [][]float32{{1, 0, 0, 0}}, K: 5, Lang: "rust"})
	if err != nil {
		t.Fatalf("indexed filter search: %v", err)
	}
	if len(hits) != 0 {
		t.Fatalf("lang=rust matched %+v", hits)
	}

	// Unit lookup picks the enclosing unit.
	u, err := b.UnitAt("pkg/quantize.go", 15)
	if err != nil {
		t.Fatalf("unit at: %v", err)
	}
	if u.Name != "quantizeEmbeddings" {
		t.Fatalf("unit at 15 = %+v", u)
	}

	// Replacing a file must not leave stale rows behind.
	if err := b.ReplaceUnits("pkg/quantize.go", nil); err != nil {
		t.Fatalf("delete: %v", err)
	}
	hits, err = b.Search(Query{Text: "quantize", K: 5})
	if err != nil {
		t.Fatalf("search after delete: %v", err)
	}
	for _, h := range hits {
		if h.Path == "pkg/quantize.go" {
			t.Fatalf("stale row survived: %+v", hits)
		}
	}
}

func TestReshapeAndQuote(t *testing.T) {
	got := reshape([]float32{1, 2, 3, 4, 5, 6}, 3)
	if len(got) != 2 || len(got[0]) != 3 || got[1][2] != 6 {
		t.Fatalf("reshape = %v", got)
	}
	if q := quote("a'b"); q != "'a''b'" {
		t.Fatalf("quote = %s", q)
	}
	q := Query{PathPrefix: "internal/", Lang: "go"}
	f := q.filter()
	if !strings.Contains(f, "path >= 'internal/'") || !strings.Contains(f, "lang = 'go'") {
		t.Fatalf("filter = %s", f)
	}
	if (Query{}).filter() != "" {
		t.Fatal("empty query must not emit a filter")
	}
}

// The filter is answered by secondary indexes, so every filtered column needs
// one, and the two that hold a handful of values need the bitmap layout.
func TestIndexSpecsAnswerTheFilter(t *testing.T) {
	secondary := map[string]*sdk.IndexInfo{}
	for _, ix := range indexSpecs(1024) {
		if ix.info.IndexType == sdk.IndexTypeSecondary {
			secondary[ix.info.TargetName] = ix.info
		}
		if ix.info.IndexType == sdk.IndexTypePLAID {
			// The centroid budget has to reach the engine: with the engine's own
			// choice a full pass retrains tens of thousands of centroids.
			if got := ix.info.Params["n_centroids"]; got != "1024" {
				t.Fatalf("plaid n_centroids = %q, want 1024", got)
			}
		}
	}
	for _, column := range []string{ColPath, ColLang, ColKind} {
		if secondary[column] == nil {
			t.Fatalf("%s carries no secondary index: %+v", column, secondary)
		}
	}
	for _, column := range []string{ColLang, ColKind} {
		if got := secondary[column].Params["cardinality"]; got != "low" {
			t.Fatalf("%s cardinality = %q, want low", column, got)
		}
	}
	if got := secondary[ColPath].Params["cardinality"]; got != "" {
		t.Fatalf("path cardinality = %q: a path is unique per file", got)
	}
}

func TestFusionParams(t *testing.T) {
	b := &Backend{cfg: Config{Fusion: FusionWeightedSum}}
	p := b.fusionParams(Query{Alpha: 0.75})
	if p["weights"] != "0.7500,0.2500" {
		t.Fatalf("weights = %v", p["weights"])
	}
	b.cfg.Fusion = FusionRRF
	if p := b.fusionParams(Query{Alpha: 0.75}); p != nil {
		t.Fatalf("rrf params = %v", p)
	}
	if b.fusion(Query{}) != FusionRRF {
		t.Fatal("fusion method not honoured")
	}
}

func splitHostPort(addr string) (string, int) {
	host, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		return addr, 23817
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return host, 23817
	}
	return host, port
}
