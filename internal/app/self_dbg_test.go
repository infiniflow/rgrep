package app_test

import (
	"fmt"
	"testing"

	"github.com/infiniflow/rgrep/internal/app"
	"github.com/infiniflow/rgrep/internal/config"
	"github.com/infiniflow/rgrep/internal/infinity"
)

func chunk(vec []float32, dim int) [][]float32 {
	n := len(vec) / dim
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		out[i] = vec[i*dim : (i+1)*dim]
	}
	return out
}

func TestDebugSelfQuery(t *testing.T) {
	cfg, err := config.Load("/tmp/rgrep-home")
	if err != nil {
		t.Fatal(err)
	}
	a, err := app.Open(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()

	u, err := a.Eng.UnitAt("internal/cli/cli.go", 120)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("unit %s:%d %s (text %d bytes)\n", u.Path, u.StartLine, u.Name, len(u.Text))
	dim := a.Emb.Dim()

	single, err := a.Emb.EncodeDocs([]string{u.Text})
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("single encode: %d bytes -> %d vectors\n", len(single[0]), len(single[0])/dim)

	batch, err := a.Emb.EncodeDocs([]string{
		u.Text, "package main", "func f() {}", "// comment", "type T struct{}",
		"import os", "return nil", "var x = 1",
	})
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("batch encode: %d bytes -> %d vectors\n", len(batch[0]), len(batch[0])/dim)

	show := func(label string, flat []float32) {
		hits, err := a.Eng.Search(infinity.Query{Vectors: chunk(flat, dim), K: 3})
		if err != nil {
			t.Fatal(err)
		}
		fmt.Println(label)
		for i, h := range hits {
			fmt.Printf("  %d. %s:%d %s %.4f\n", i+1, h.Path, h.StartLine, h.Name, h.Score)
		}
	}

	show("query from single encode", single[0])
	show("query from batch encode", batch[0])

	// Compare the two encodings of the same text, per token vector.
	diff := 0
	maxAbs := float32(0)
	if len(single[0]) == len(batch[0]) {
		for i := range single[0] {
			d := single[0][i] - batch[0][i]
			if d < 0 {
				d = -d
			}
			if d > maxAbs {
				maxAbs = d
			}
			if single[0][i] != batch[0][i] {
				diff++
			}
		}
	}
	fmt.Printf("single vs batch: len %d/%d, differing floats %d, max abs diff %.6f\n",
		len(single[0]), len(batch[0]), diff, maxAbs)

	// Mean cosine between corresponding token vectors.
	if len(single[0]) == len(batch[0]) {
		var sum float64
		n := 0
		for t := 0; t+dim <= len(single[0]); t += dim {
			var dot float64
			for d := 0; d < dim; d++ {
				dot += float64(single[0][t+d] * batch[0][t+d])
			}
			sum += dot
			n++
		}
		if n > 0 {
			fmt.Printf("mean per-token cosine single vs batch: %.6f\n", sum/float64(n))
		}
	}
}
