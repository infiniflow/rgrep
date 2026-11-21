package store

import (
	"path/filepath"
	"testing"
)

func TestCatalogRoundTrip(t *testing.T) {
	dir := t.TempDir()
	st, err := Open(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	if err := st.PutFile("a.go", FileState{Hash: "abc", MtimeNs: 1, Size: 10, Lang: "go", Units: 2}); err != nil {
		t.Fatal(err)
	}
	if err := st.PutFile("b.go", FileState{Hash: "def", MtimeNs: 2, Size: 20, Lang: "go", Units: 3}); err != nil {
		t.Fatal(err)
	}

	files, err := st.Files()
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 || files["a.go"].Units != 2 || files["b.go"].Hash != "def" {
		t.Fatalf("files = %v", files)
	}

	// Upserting a path replaces its row instead of adding one.
	if err := st.PutFile("a.go", FileState{Hash: "xyz", MtimeNs: 3, Size: 11, Lang: "go", Units: 1}); err != nil {
		t.Fatal(err)
	}
	files, err = st.Files()
	if err != nil || len(files) != 2 || files["a.go"].Hash != "xyz" || files["a.go"].Units != 1 {
		t.Fatalf("after upsert files = %v, err = %v", files, err)
	}

	// Touch refreshes the fingerprint without changing the unit count.
	if err := st.TouchFile("a.go", FileState{Hash: "xyz", MtimeNs: 9, Size: 12, Lang: "go", Units: 1}); err != nil {
		t.Fatal(err)
	}
	files, _ = st.Files()
	if files["a.go"].MtimeNs != 9 || files["a.go"].Size != 12 {
		t.Fatalf("after touch = %+v", files["a.go"])
	}

	f, u, err := st.Stats()
	if err != nil || f != 2 || u != 4 {
		t.Fatalf("stats files=%d units=%d err=%v", f, u, err)
	}

	if err := st.DeleteFile("b.go"); err != nil {
		t.Fatal(err)
	}
	f, u, err = st.Stats()
	if err != nil || f != 1 || u != 1 {
		t.Fatalf("after delete files=%d units=%d err=%v", f, u, err)
	}
}

func TestMeta(t *testing.T) {
	st, err := Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	if got := st.GetMeta("embedder"); got != "" {
		t.Fatalf("missing key = %q", got)
	}
	if err := st.SetMeta("embedder", "onnx:x"); err != nil {
		t.Fatal(err)
	}
	if err := st.SetMeta("embedder", "onnx:y"); err != nil {
		t.Fatal(err)
	}
	if got := st.GetMeta("embedder"); got != "onnx:y" {
		t.Fatalf("meta = %q", got)
	}
}

func TestUnitNVec(t *testing.T) {
	u := Unit{Dim: 4, Vec: make([]float32, 12)}
	if got := u.NVec(); got != 3 {
		t.Fatalf("nvec = %d", got)
	}
	if got := (&Unit{}).NVec(); got != 0 {
		t.Fatalf("empty nvec = %d", got)
	}
}
