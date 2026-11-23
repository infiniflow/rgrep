package embed

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// benchEncoder builds an encoder against the local model checkout, skipping the
// benchmark on a machine that does not have one.
func benchEncoder(tb testing.TB, sessions int) *ONNX {
	tb.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		tb.Skip("no home directory")
	}
	modelDir := filepath.Join(home, "codebase", "workspace", "LateOn-Code-edge")
	libDir := filepath.Join(home, "codebase", "workspace", "rgrep", "bin", "onnxruntime", "lib")
	for _, p := range []string{
		filepath.Join(modelDir, "model_int8.onnx"),
		filepath.Join(modelDir, "tokenizer.json"),
		libDir,
	} {
		if _, err := os.Stat(p); err != nil {
			tb.Skipf("missing %s", p)
		}
	}
	o, err := NewONNX(ONNXOptions{
		ModelPath:     filepath.Join(modelDir, "model_int8.onnx"),
		TokenizerPath: filepath.Join(modelDir, "tokenizer.json"),
		ORTLibDir:     libDir,
		Dim:           48,
		DocLen:        128,
		QueryLen:      256,
		ClsID:         50281,
		SepID:         50282,
		PadID:         50283,
		PrefixQ:       50368,
		PrefixD:       50369,
		BatchSize:     8,
		Sessions:      sessions,
	})
	if err != nil {
		tb.Fatalf("encoder: %v", err)
	}
	tb.Cleanup(func() { _ = o.Close() })
	return o
}

// benchTexts is `files` files' worth of units: 12 texts each, of the size a
// code unit produces (a signature plus body, 300-900 characters).
func benchTexts(files int) []string {
	base := "func (s *Service) HandleRequest(ctx context.Context, req *Request) (*Response, error) { " +
		"if req == nil { return nil, errors.New(\"nil request\") } " +
		"rows, err := s.store.Query(ctx, req.Filter) " +
		"if err != nil { return nil, fmt.Errorf(\"query: %w\", err) } " +
		"defer rows.Close() // the caller owns the cursor"
	texts := make([]string, 0, 12*files)
	for i := 0; i < 12*files; i++ {
		texts = append(texts, fmt.Sprintf("// handler %d\n%s\n service%d", i, base, i))
	}
	return texts
}

// BenchmarkEncodeDocs measures one file's encode for several session counts -
// the unit rgrep pays per changed file while indexing.
func BenchmarkEncodeDocs(b *testing.B) {
	for _, sessions := range []int{1, 2, 4, 6, 8, 10} {
		b.Run(fmt.Sprintf("sessions=%02d", sessions), func(b *testing.B) {
			o := benchEncoder(b, sessions)
			texts := benchTexts(1)
			if _, err := o.EncodeDocs(texts); err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := o.EncodeDocs(texts); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			// Units per second is what the indexer sees.
			b.ReportMetric(float64(len(texts))*float64(b.N)/b.Elapsed().Seconds(), "units/s")
		})
	}
}

// BenchmarkEncodeBatch is the same work handed over twenty files at a time: the
// shape an indexer gets if it batches across files instead of encoding each one
// on its own.
func BenchmarkEncodeBatch(b *testing.B) {
	for _, sessions := range []int{1, 2, 4, 6, 8, 10} {
		b.Run(fmt.Sprintf("sessions=%02d", sessions), func(b *testing.B) {
			o := benchEncoder(b, sessions)
			texts := benchTexts(20)
			if _, err := o.EncodeDocs(texts); err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := o.EncodeDocs(texts); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(len(texts))*float64(b.N)/b.Elapsed().Seconds(), "units/s")
		})
	}
}

// BenchmarkEncodeQuery is the per-search cost: a single text, so the pool has
// nothing to parallelise.
func BenchmarkEncodeQuery(b *testing.B) {
	for _, sessions := range []int{1, 10} {
		b.Run(fmt.Sprintf("sessions=%02d", sessions), func(b *testing.B) {
			o := benchEncoder(b, sessions)
			if _, err := o.EncodeQuery("how does the engine publish a chunk"); err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := o.EncodeQuery("how does the engine publish a chunk"); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
