package embed

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	ort "github.com/infiniflow/onnxruntime_go"

	"github.com/infiniflow/rgrep/internal/native"
)

// ONNXOptions configures the LateOn-Code-edge encoder.
type ONNXOptions struct {
	ModelPath     string
	TokenizerPath string
	ORTLibDir     string
	Dim           int
	DocLen        int
	QueryLen      int
	ClsID         int32
	SepID         int32
	PadID         int32
	PrefixQ       int32
	PrefixD       int32
	BatchSize     int
}

// ONNX runs the ColBERT model through onnxruntime.
type ONNX struct {
	opt  ONNXOptions
	sess *ort.DynamicAdvancedSession
	mu   sync.Mutex
}

var (
	ortOnce sync.Once
	ortErr  error
)

// FindORTLib locates a libonnxruntime shared library in dir.
func FindORTLib(dir string) (string, error) {
	if dir == "" {
		return "", fmt.Errorf("onnxruntime lib dir not configured")
	}
	patterns := []string{"libonnxruntime*.dylib", "libonnxruntime*.so*", "onnxruntime*.dll"}
	var found []string
	for _, p := range patterns {
		m, _ := filepath.Glob(filepath.Join(dir, p))
		found = append(found, m...)
	}
	if len(found) == 0 {
		return "", fmt.Errorf("no onnxruntime shared library in %s", dir)
	}
	sort.Sort(sort.Reverse(sort.StringSlice(found)))
	return found[0], nil
}

// NewONNX opens the model. It fails if the model or the runtime is missing.
func NewONNX(opt ONNXOptions) (*ONNX, error) {
	if opt.BatchSize <= 0 {
		opt.BatchSize = 8
	}
	if opt.ClsID == 0 {
		opt.ClsID = 50281
	}
	if opt.SepID == 0 {
		opt.SepID = 50282
	}
	if opt.PadID == 0 {
		opt.PadID = 50283
	}
	if _, err := os.Stat(opt.ModelPath); err != nil {
		return nil, fmt.Errorf("model: %w", err)
	}
	if _, err := os.Stat(opt.TokenizerPath); err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}

	lib, err := FindORTLib(opt.ORTLibDir)
	if err != nil {
		return nil, err
	}

	ortOnce.Do(func() {
		ort.SetSharedLibraryPath(lib)
		ortErr = ort.InitializeEnvironment()
	})
	if ortErr != nil {
		return nil, fmt.Errorf("onnxruntime init: %w", ortErr)
	}

	so, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer so.Destroy()

	sess, err := ort.NewDynamicAdvancedSession(opt.ModelPath,
		[]string{"input_ids", "attention_mask"}, []string{"output"}, so)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	return &ONNX{opt: opt, sess: sess}, nil
}

// Dim implements Embedder.
func (o *ONNX) Dim() int { return o.opt.Dim }

// Name implements Embedder.
func (o *ONNX) Name() string { return "lateon-code-edge:" + filepath.Base(o.opt.ModelPath) }

// Close implements Embedder.
func (o *ONNX) Close() error { return o.sess.Destroy() }

// EncodeDocs implements Embedder.
func (o *ONNX) EncodeDocs(texts []string) ([][]float32, error) {
	return o.encode(texts, o.opt.DocLen, o.opt.PrefixD)
}

// EncodeQuery implements Embedder.
func (o *ONNX) EncodeQuery(text string) ([]float32, error) {
	vs, err := o.encode([]string{text}, o.opt.QueryLen, o.opt.PrefixQ)
	if err != nil {
		return nil, err
	}
	if len(vs) == 0 {
		return nil, nil
	}
	return vs[0], nil
}

// encode tokenizes and runs the model.
//
// Texts are batched only with texts of the *same* token length. The int8
// export does not make padding neutral — a text encoded next to a longer one
// comes out with token vectors about 0.997 cosine-similar to the same text
// encoded alone (measured), which is enough to swamp a code search where
// unrelated units already sit near 0.99. Batching by exact length keeps the
// result independent of who else is in the batch, so a stored document and a
// query at retrieval time see identical vectors.
func (o *ONNX) encode(texts []string, maxLen int, prefix int32) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	groups := make(map[int][]encRow, len(texts))
	for i, t := range texts {
		toks, err := native.Tokenize(o.opt.TokenizerPath, t)
		if err != nil {
			return nil, err
		}
		if maxLen > 3 && len(toks) > maxLen-2 {
			toks = toks[:maxLen-2]
		}
		ids := make([]int32, 0, len(toks)+2)
		ids = append(ids, o.opt.ClsID, prefix)
		ids = append(ids, toks...)
		ids = append(ids, o.opt.SepID)
		groups[len(ids)] = append(groups[len(ids)], encRow{idx: i, ids: ids, nToks: len(toks)})
	}

	out := make([][]float32, len(texts))
	bs := o.opt.BatchSize
	o.mu.Lock()
	defer o.mu.Unlock()

	for tLen, rows := range groups {
		for start := 0; start < len(rows); start += bs {
			end := start + bs
			if end > len(rows) {
				end = len(rows)
			}
			if err := o.runBatch(rows[start:end], tLen, out); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

// encRow is one tokenized text waiting for a forward pass.
type encRow struct {
	idx   int
	ids   []int32
	nToks int
}

// runBatch encodes rows that all have exactly tLen tokens, so there is no
// padding and the result does not depend on the other rows in the batch.
func (o *ONNX) runBatch(rows []encRow, tLen int, out [][]float32) error {
	n := len(rows)
	ids := make([]int64, n*tLen)
	mask := make([]int64, n*tLen)
	for b := range rows {
		for j, id := range rows[b].ids {
			ids[b*tLen+j] = int64(id)
			mask[b*tLen+j] = 1
		}
	}

	inIDs, err := ort.NewTensor(ort.NewShape(int64(n), int64(tLen)), ids)
	if err != nil {
		return err
	}
	defer inIDs.Destroy()
	inMask, err := ort.NewTensor(ort.NewShape(int64(n), int64(tLen)), mask)
	if err != nil {
		return err
	}
	defer inMask.Destroy()
	outT, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(n), int64(tLen), int64(o.opt.Dim)))
	if err != nil {
		return err
	}
	defer outT.Destroy()

	if err := o.sess.Run([]ort.Value{inIDs, inMask}, []ort.Value{outT}); err != nil {
		return fmt.Errorf("session run: %w", err)
	}
	data := outT.GetData()
	for b := range rows {
		base := b * tLen * o.opt.Dim
		// Drop [CLS], the [Q]/[D] prefix and [SEP]: keep content tokens only.
		lo := 2
		hi := 2 + rows[b].nToks
		flat := make([]float32, 0, (hi-lo)*o.opt.Dim)
		for j := lo; j < hi; j++ {
			off := base + j*o.opt.Dim
			if off+o.opt.Dim > len(data) {
				break
			}
			flat = append(flat, data[off:off+o.opt.Dim]...)
		}
		normalizeRows(flat, o.opt.Dim)
		out[rows[b].idx] = flat
	}
	return nil
}

// DecodeIDs is a debugging helper: tokenize without running the model.
func (o *ONNX) DecodeIDs(text string) ([]int32, error) {
	return native.Tokenize(o.opt.TokenizerPath, strings.TrimSpace(text))
}
