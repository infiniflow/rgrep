package app_test

import (
	"fmt"
	"testing"

	"github.com/infiniflow/rgrep/internal/app"
	"github.com/infiniflow/rgrep/internal/config"
)

// TestDebugPadding checks whether the encoder's output depends on the padding
// that a batch introduces, or on the batch itself.
func TestDebugPadding(t *testing.T) {
	cfg, err := config.Load("/tmp/rgrep-home")
	if err != nil {
		t.Fatal(err)
	}
	a, err := app.Open(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	dim := a.Emb.Dim()

	u, err := a.Eng.UnitAt("internal/cli/cli.go", 120)
	if err != nil {
		t.Fatal(err)
	}
	text := u.Text

	same, err := a.Emb.EncodeDocs([]string{text, text, text, text})
	if err != nil {
		t.Fatal(err)
	}
	long, err := a.Emb.EncodeDocs([]string{text, text + "\n// tail line that makes this text longer", "package main"})
	if err != nil {
		t.Fatal(err)
	}

	maxdiff := func(x, y []float32) float32 {
		var m float32
		for i := range x {
			d := x[i] - y[i]
			if d < 0 {
				d = -d
			}
			if d > m {
				m = d
			}
		}
		return m
	}
	fmt.Printf("same-length batch vs padded batch: max diff %.6f (lens %d/%d)\n",
		maxdiff(same[0], long[0]), len(same[0]), len(long[0]))

	// And two identical texts in the same batch.
	fmt.Printf("within same-length batch: max diff %.6f\n", maxdiff(same[0], same[1]))

	// Cosine between same[0] and long[0].
	var dot float64
	for i := range same[0] {
		dot += float64(same[0][i] * long[0][i])
	}
	n := len(same[0]) / dim
	fmt.Printf("mean token cosine %.6f\n", dot/float64(n))
	_ = dim
}
