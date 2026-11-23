// Command rgrep-eval measures retrieval quality of an existing index.
//
// It answers the only question worth asking about a recall backend: given the
// repository's own prose (docstrings, Go doc comments, JSDoc) or its
// identifiers, does the right unit come back, and how high? The ground truth is
// (path, name) taken from the source tree, not from the index, so a change in
// fusion or in the vector backend shows up as a number instead of an opinion.
//
// The index itself is built by `rgrep index`; this tool only reads it.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/infiniflow/rgrep/internal/app"
	"github.com/infiniflow/rgrep/internal/config"
	"github.com/infiniflow/rgrep/internal/search"
)

type query struct {
	Set   string `json:"set"`
	Query string `json:"query"`
	Path  string `json:"path"`
	Name  string `json:"name"`
}

type stats struct {
	Set        string
	Total      int
	Hits       int
	Reciprocal float64
	Latency    []time.Duration
}

func (s *stats) add(rank int, elapsed time.Duration) {
	s.Total++
	s.Latency = append(s.Latency, elapsed)
	if rank > 0 {
		s.Hits++
		s.Reciprocal += 1 / float64(rank)
	}
}

func (s *stats) report(k int) map[string]any {
	sort.Slice(s.Latency, func(i, j int) bool { return s.Latency[i] < s.Latency[j] })
	pick := func(percentile float64) time.Duration {
		if len(s.Latency) == 0 {
			return 0
		}
		index := int(percentile * float64(len(s.Latency)-1))
		return s.Latency[index]
	}
	out := map[string]any{
		"queries":                 s.Total,
		"recall@" + fmt.Sprint(k): float64(s.Hits) / float64(max(s.Total, 1)),
		"mrr":                     s.Reciprocal / float64(max(s.Total, 1)),
		"p50_ms":                  float64(pick(0.50).Microseconds()) / 1000,
		"p95_ms":                  float64(pick(0.95).Microseconds()) / 1000,
	}
	return out
}

func main() {
	home := flag.String("home", "", "rgrep data directory (the index to evaluate)")
	queriesPath := flag.String("queries", "", "queries.jsonl produced by scripts/eval_queries.py")
	k := flag.Int("k", 10, "cut-off for recall@k")
	alpha := flag.Float64("alpha", 0.6, "semantic weight of the engine-side fusion")
	only := flag.String("set", "", "evaluate only this query set (nl or symbol)")
	flag.Parse()

	if *home == "" || *queriesPath == "" {
		fmt.Fprintln(os.Stderr, "usage: rgrep-eval -home DIR -queries FILE [-k 10] [-alpha 0.6] [-set nl]")
		os.Exit(2)
	}

	cfg, err := config.Load(*home)
	if err != nil {
		fmt.Fprintln(os.Stderr, "config:", err)
		os.Exit(1)
	}
	a, err := app.Open(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer a.Close()

	queries, err := readQueries(*queriesPath, *only)
	if err != nil {
		fmt.Fprintln(os.Stderr, "queries:", err)
		os.Exit(1)
	}
	if len(queries) == 0 {
		fmt.Fprintln(os.Stderr, "no queries")
		os.Exit(1)
	}

	searcher := a.Searcher()
	bySet := map[string]*stats{}
	for _, q := range queries {
		started := time.Now()
		results, err := searcher.Search(q.Query, search.Options{K: *k, Alpha: *alpha})
		elapsed := time.Since(started)
		if err != nil {
			fmt.Fprintf(os.Stderr, "query %q: %v\n", q.Query, err)
			continue
		}
		rank := rankOf(results, q)
		entry := bySet[q.Set]
		if entry == nil {
			entry = &stats{Set: q.Set}
			bySet[q.Set] = entry
		}
		entry.add(rank, elapsed)
	}

	summary := map[string]any{
		"embedding": cfg.Mode(),
		"encoder":   a.Note,
		"k":         *k,
		"alpha":     *alpha,
	}
	for set, entry := range bySet {
		summary[set] = entry.report(*k)
	}
	encoded, _ := json.MarshalIndent(summary, "", "  ")
	fmt.Println(string(encoded))
}

func readQueries(path, only string) ([]query, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var out []query
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var q query
		if err := json.Unmarshal([]byte(line), &q); err != nil {
			return nil, fmt.Errorf("parse %q: %w", line, err)
		}
		if only != "" && q.Set != only {
			continue
		}
		out = append(out, q)
	}
	return out, scanner.Err()
}

// rankOf returns the 1-based position of the ground-truth unit, or 0.
//
// Paths are compared by suffix because the index may have been built with an
// absolute root and the query set with a relative one; names must match exactly,
// which is what makes a hit unambiguous even when a file has two same-named
// methods on different classes.
func rankOf(results []search.Result, q query) int {
	want := filepath.ToSlash(q.Path)
	for index, result := range results {
		if result.Name != q.Name {
			continue
		}
		got := filepath.ToSlash(result.Path)
		if got == want || strings.HasSuffix(got, "/"+want) {
			return index + 1
		}
	}
	return 0
}
