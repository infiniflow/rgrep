// Package search is the thin layer between an agent's query and Infinity.
//
// Recall, scoring and fusion all happen in the engine (see package infinity):
// the vector branch (PLAID multi-vector MaxSim or HNSW dense) and the BM25
// full-text branch are combined there with `FUSION('weighted_sum')` or
// `FUSION('rrf')`. Which vector operator is emitted depends only on the
// encoder's shape (Embedder.Multi), not on a separate mode flag. What is left
// here is the part that is about code rather than about vectors: penalising
// test files, boosting a unit whose name is exactly what was typed, and
// keeping one file from filling the result list.
package search

import (
	"sort"
	"strings"
	"unicode"

	"github.com/infiniflow/rgrep/internal/embed"
	"github.com/infiniflow/rgrep/internal/index"
	"github.com/infiniflow/rgrep/internal/infinity"
)

// Options controls one query.
type Options struct {
	K          int
	Alpha      float64 // weight of the semantic branch (weighted_sum fusion)
	PathPrefix string
	Lang       string
	Kind       string
	NoTests    bool
	// Candidates is how many fused rows are fetched before re-ranking.
	Candidates int
	// Engine carries the recall operator's knobs through to the engine.
	Engine map[string]string
}

// Result is one ranked code unit.
type Result struct {
	Path    string  `json:"path"`
	Lang    string  `json:"lang"`
	Kind    string  `json:"kind"`
	Name    string  `json:"name"`
	Parent  string  `json:"parent"`
	StartLn int     `json:"start_line"`
	EndLn   int     `json:"end_line"`
	Score   float64 `json:"score"`
	// Rank is the engine's rank before rgrep's re-ranking (1 = best).
	Rank int    `json:"rank"`
	Code string `json:"code,omitempty"`
}

// Retriever is the engine side: one hybrid SELECT.
type Retriever interface {
	Search(q infinity.Query) ([]infinity.Hit, error)
}

// Searcher encodes the query, asks Infinity for candidates, then re-ranks.
type Searcher struct {
	r   Retriever
	emb embed.Embedder
}

// New builds a searcher.
func New(r Retriever, emb embed.Embedder) *Searcher {
	return &Searcher{r: r, emb: emb}
}

// Search executes a hybrid query.
func (s *Searcher) Search(query string, opt Options) ([]Result, error) {
	if opt.K <= 0 {
		opt.K = 10
	}
	if opt.Alpha <= 0 {
		opt.Alpha = 0.6
	}
	if opt.Candidates <= 0 {
		opt.Candidates = 200
	}
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, nil
	}

	dim := s.emb.Dim()
	qvec, err := s.emb.EncodeQuery(query)
	if err != nil {
		return nil, err
	}

	q := infinity.Query{
		Text:       FullText(query),
		K:          opt.Candidates,
		Alpha:      opt.Alpha,
		PathPrefix: opt.PathPrefix,
		Lang:       opt.Lang,
		Kind:       opt.Kind,
		Options:    opt.Engine,
	}
	// The encoder decides the query shape: per-token vectors go to the PLAID
	// branch, one pooled vector goes to the HNSW branch.
	if s.emb.Multi() {
		q.Vectors = chunk(qvec, dim)
	} else {
		q.Vector = qvec
	}

	hits, err := s.r.Search(q)
	if err != nil {
		return nil, err
	}
	if len(hits) == 0 {
		return nil, nil
	}

	qTokens := tokenSet(query)
	out := make([]Result, 0, len(hits))
	for i, h := range hits {
		if opt.NoTests && isTestPath(h.Path) {
			continue
		}
		r := Result{
			Path: h.Path, Lang: h.Lang, Kind: h.Kind, Name: h.Name, Parent: h.Parent,
			StartLn: h.StartLine, EndLn: h.EndLine,
			Score: h.Score, Rank: i + 1, Code: h.Code,
		}
		r.Score *= boost(&r, qTokens)
		out = append(out, r)
	}
	if len(out) == 0 {
		return nil, nil
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].Score == out[j].Score {
			return out[i].Rank < out[j].Rank
		}
		return out[i].Score > out[j].Score
	})
	return collapse(out, opt.K), nil
}

// collapse keeps at most two units per file so one big class cannot flood the
// result list, then trims to k.
func collapse(in []Result, k int) []Result {
	perFile := make(map[string]int, len(in))
	out := make([]Result, 0, k)
	for _, r := range in {
		if perFile[r.Path] >= 2 {
			continue
		}
		perFile[r.Path]++
		out = append(out, r)
		if len(out) >= k {
			break
		}
	}
	return out
}

func isTestPath(p string) bool {
	lp := strings.ToLower(p)
	return strings.Contains(lp, "/test/") || strings.Contains(lp, "/tests/") ||
		strings.Contains(lp, "_test.") || strings.Contains(lp, ".test.") ||
		strings.Contains(lp, ".spec.") || strings.Contains(lp, "/spec/") ||
		strings.HasSuffix(lp, "_test.go")
}

func boost(r *Result, qTokens map[string]bool) float64 {
	f := 1.0
	name := strings.ToLower(r.Name)
	if name != "" && qTokens[name] {
		f += 0.55
	} else if name != "" {
		for t := range qTokens {
			if len(t) > 2 && strings.HasPrefix(name, t) {
				f += 0.2
				break
			}
		}
	}
	switch r.Kind {
	case "function", "method":
		f += 0.06
	case "class", "struct", "interface", "trait":
		f += 0.04
	case "raw":
		f -= 0.05
	}
	lp := strings.ToLower(r.Path)
	if isTestPath(lp) {
		f *= 0.55
	}
	if strings.Contains(lp, "example") || strings.Contains(lp, "doc/") || strings.Contains(lp, "docs/") {
		f *= 0.85
	}
	if r.Parent != "" {
		f += 0.02
	}
	return f
}

func tokenSet(s string) map[string]bool {
	out := make(map[string]bool)
	for _, t := range strings.FieldsFunc(s, func(r rune) bool {
		return !(unicode.IsLetter(r) || unicode.IsDigit(r))
	}) {
		if len(t) >= 2 {
			out[strings.ToLower(t)] = true
		}
	}
	return out
}

// FullText turns free text into the string the engine tokenises: identifiers
// are pre-split (`parse_arguments` → `parse arguments`) and the terms are
// space-separated. Infinity's default operator is OR, so any term may match,
// which is what a symbol lookup wants.
func FullText(q string) string {
	terms := index.SplitIdents(q)
	seen := make(map[string]bool)
	var parts []string
	for _, t := range strings.Fields(terms) {
		t = strings.ToLower(t)
		// The engine parses the text branch as an expression: a bare `and`,
		// `or` or `not` from prose ("this does not resolve") turns the whole
		// query into "incomplete boolean expression" and loses every hit, so
		// operator words never reach it.
		if len(t) < 2 || seen[t] || isOperator(t) {
			continue
		}
		seen[t] = true
		parts = append(parts, t)
		if len(parts) >= 16 {
			break
		}
	}
	return strings.Join(parts, " ")
}

// isOperator reports words the engine's full-text parser treats as operators.
func isOperator(term string) bool {
	switch term {
	case "and", "or", "not":
		return true
	default:
		return false
	}
}

// chunk splits a flat token buffer into one slice per token vector.
func chunk(vec []float32, dim int) [][]float32 {
	if dim <= 0 || len(vec) == 0 {
		return nil
	}
	n := len(vec) / dim
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		out[i] = vec[i*dim : (i+1)*dim]
	}
	return out
}
