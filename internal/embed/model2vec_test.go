package embed

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// golden holds vectors produced by the Python reference implementation
// (`model2vec.StaticModel.encode`) for the same texts, so the Rust port is
// checked against the format's own definition rather than against itself.
type golden struct {
	Model   string      `json:"model"`
	Texts   []string    `json:"texts"`
	Vectors [][]float32 `json:"vectors"`
}

func loadGolden(t *testing.T) golden {
	t.Helper()
	path := filepath.Join("testdata", "potion-code-golden.json")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var g golden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return g
}

// The model itself is several tens of megabytes and not in the repo, so this
// runs only when a checkout is pointed at.
func TestModel2VecMatchesThePythonReference(t *testing.T) {
	dir := os.Getenv("RGREP_M2V_MODEL")
	if dir == "" {
		t.Skip("set RGREP_M2V_MODEL to a potion-code-16M-v2 checkout to run")
	}
	g := loadGolden(t)

	enc, err := NewModel2Vec(dir)
	if err != nil {
		t.Fatalf("open model: %v", err)
	}
	defer enc.Close()

	if enc.Multi() {
		t.Fatal("a static model must report Multi() == false")
	}
	if enc.Dim() != len(g.Vectors[0]) {
		t.Fatalf("dim = %d, golden has %d", enc.Dim(), len(g.Vectors[0]))
	}

	got, err := enc.EncodeDocs(g.Texts)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	for i := range got {
		if cos := cosine(got[i], g.Vectors[i]); cos < 0.99999 {
			t.Errorf("text %d: cosine with the reference is %.6f, want >= 0.99999", i, cos)
		}
		if maxAbsDiff(got[i], g.Vectors[i]) > 5e-4 {
			t.Errorf("text %d: max component difference %.2e exceeds 5e-4", i, maxAbsDiff(got[i], g.Vectors[i]))
		}
	}
}

func TestModel2VecMissingDirIsAnError(t *testing.T) {
	if _, err := NewModel2Vec(filepath.Join(t.TempDir(), "nope")); err == nil {
		t.Fatal("expected an error for a missing model directory")
	}
	if _, err := NewModel2Vec(""); err == nil {
		t.Fatal("expected an error for an empty model directory")
	}
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func maxAbsDiff(a, b []float32) float64 {
	var worst float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > worst {
			worst = d
		}
	}
	return worst
}
