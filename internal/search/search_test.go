package search

import (
	"strings"
	"testing"

	"github.com/infiniflow/rgrep/internal/embed"
	"github.com/infiniflow/rgrep/internal/infinity"
)

// stubEmbedder returns a fixed-width token vector per token, so a query of n
// tokens produces n vectors.
type stubEmbedder struct {
	dim    int
	tokens int
}

func (e *stubEmbedder) Dim() int { return e.dim }

func (e *stubEmbedder) EncodeDocs(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = make([]float32, e.dim)
		out[i][0] = 1
	}
	return out, nil
}

func (e *stubEmbedder) EncodeQuery(string) ([]float32, error) {
	flat := make([]float32, e.tokens*e.dim)
	for i := 0; i < e.tokens; i++ {
		flat[i*e.dim] = 1
	}
	return flat, nil
}

func (e *stubEmbedder) Name() string { return "stub" }
func (e *stubEmbedder) Close() error { return nil }

type stubRetriever struct {
	got  infinity.Query
	hits []infinity.Hit
}

func (r *stubRetriever) Search(q infinity.Query) ([]infinity.Hit, error) {
	r.got = q
	return r.hits, nil
}

func TestSearchPushesQueryIntoTheEngine(t *testing.T) {
	r := &stubRetriever{hits: []infinity.Hit{{Path: "a.go", StartLine: 1, EndLine: 2, Score: 1}}}
	s := New(r, &stubEmbedder{dim: 4, tokens: 3})

	if _, err := s.Search("parse_arguments", Options{K: 5, Alpha: 0.7, Lang: "go"}); err != nil {
		t.Fatal(err)
	}
	if len(r.got.Vectors) != 3 {
		t.Fatalf("query vectors = %d, want 3", len(r.got.Vectors))
	}
	if r.got.Text == "" {
		t.Fatal("full-text side of the query is empty")
	}
	if r.got.Alpha != 0.7 || r.got.Lang != "go" {
		t.Fatalf("query = %+v", r.got)
	}
}

func TestFullTextSplitsIdentifiers(t *testing.T) {
	got := FullText("getUserById")
	for _, want := range []string{"getuserbyid", "get", "user", "by", "id"} {
		if !containsField(got, want) {
			t.Fatalf("%q missing %q", got, want)
		}
	}
}

func TestBoostPrefersTheDefinition(t *testing.T) {
	r := &stubRetriever{hits: []infinity.Hit{
		{Path: "a.go", Kind: "raw", Name: "unrelated", StartLine: 1, EndLine: 2, Score: 1},
		{Path: "b.go", Kind: "function", Name: "parseConfig", StartLine: 1, EndLine: 2, Score: 1},
	}}
	res, err := New(r, &stubEmbedder{dim: 4, tokens: 1}).Search("parseConfig", Options{K: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 || res[0].Name != "parseConfig" {
		t.Fatalf("results = %+v", res)
	}
}

func TestCollapseKeepsTwoPerFile(t *testing.T) {
	var hits []infinity.Hit
	for i := 0; i < 4; i++ {
		hits = append(hits, infinity.Hit{Path: "a.go", Kind: "function", StartLine: i * 10, EndLine: i*10 + 5, Score: float64(4 - i)})
	}
	hits = append(hits, infinity.Hit{Path: "b.go", Kind: "function", StartLine: 1, EndLine: 2, Score: 0.1})

	res, err := New(&stubRetriever{hits: hits}, &stubEmbedder{dim: 4, tokens: 1}).Search("x", Options{K: 4})
	if err != nil {
		t.Fatal(err)
	}
	perFile := map[string]int{}
	for _, r := range res {
		perFile[r.Path]++
	}
	if perFile["a.go"] > 2 {
		t.Fatalf("a.go kept %d results", perFile["a.go"])
	}
	if perFile["b.go"] != 1 {
		t.Fatalf("b.go dropped: %+v", res)
	}
}

func TestNoTestsDropsTestPaths(t *testing.T) {
	r := &stubRetriever{hits: []infinity.Hit{
		{Path: "pkg/a_test.go", Kind: "function", StartLine: 1, EndLine: 2, Score: 2},
		{Path: "pkg/a.go", Kind: "function", StartLine: 1, EndLine: 2, Score: 1},
	}}
	res, err := New(r, &stubEmbedder{dim: 4, tokens: 1}).Search("x", Options{K: 5, NoTests: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].Path != "pkg/a.go" {
		t.Fatalf("results = %+v", res)
	}
}

func containsField(s, want string) bool {
	for _, f := range strings.Fields(s) {
		if f == want {
			return true
		}
	}
	return false
}

var _ embed.Embedder = (*stubEmbedder)(nil)
