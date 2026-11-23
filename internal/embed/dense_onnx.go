package embed

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	ort "github.com/infiniflow/onnxruntime_go"

	"github.com/infiniflow/rgrep/internal/native"
)

// DenseOptions configures a single-vector encoder.
//
// The model is whatever sentence-transformers export the user points at: the
// contract is "token ids in, one pooled vector out". Both `last_hidden_state`
// (rank 3, pooled here) and `sentence_embedding` (rank 2, used as is) work, and
// the output name is discovered from the model when Output is empty.
type DenseOptions struct {
	ModelPath     string
	TokenizerPath string
	ORTLibDir     string

	// Dim is the expected embedding width. 0 infers it from the model's output
	// shape, which is the only place that number is authoritative.
	Dim int
	// MaxLen caps the token count per text (default 512).
	MaxLen int
	// Pooling is "mean" (default) or "cls", used only when the model emits
	// token-level states. Ties are broken by the model's own sentence
	// embedding when it has one.
	Pooling string
	// Output names the graph output to read; empty auto-detects.
	Output string
	// ClsID/SepID are added around the tokens. They default to 101/102 (BERT
	// family); set both to 0 for a tokenizer that already wraps its input.
	ClsID int32
	SepID int32

	BatchSize int
	// Sessions is how many forward passes run at once. 0 means one per CPU,
	// capped at 8, mirroring the ColBERT encoder.
	Sessions int
}

// DenseONNX runs a single-vector model through onnxruntime.
type DenseONNX struct {
	opt        DenseOptions
	pool       chan *ort.DynamicAdvancedSession
	inputs     []denseInput
	outputName string
	outputRank int
}

// denseInput is one graph input the encoder fills itself.
type denseInput struct {
	name string
	kind denseInputKind
}

type denseInputKind int

const (
	inputIDs denseInputKind = iota
	inputMask
	inputTokenTypes
)

// NewDenseONNX opens the model. It fails if the model or the runtime is missing.
func NewDenseONNX(opt DenseOptions) (*DenseONNX, error) {
	if opt.BatchSize <= 0 {
		opt.BatchSize = 8
	}
	if opt.MaxLen <= 0 {
		opt.MaxLen = 512
	}
	// Both zero means "the tokenizer already wraps the text".
	if !(opt.ClsID == 0 && opt.SepID == 0) {
		if opt.ClsID == 0 {
			opt.ClsID = 101
		}
		if opt.SepID == 0 {
			opt.SepID = 102
		}
	}
	if opt.Pooling == "" {
		opt.Pooling = "mean"
	}
	if opt.Pooling != "mean" && opt.Pooling != "cls" {
		return nil, fmt.Errorf("dense pooling %q: use mean or cls", opt.Pooling)
	}
	if _, err := os.Stat(opt.ModelPath); err != nil {
		return nil, fmt.Errorf("dense model: %w", err)
	}
	if _, err := os.Stat(opt.TokenizerPath); err != nil {
		return nil, fmt.Errorf("dense tokenizer: %w", err)
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

	ins, outs, err := ort.GetInputOutputInfo(opt.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("inspect dense model: %w", err)
	}
	inputs, err := mapDenseInputs(ins)
	if err != nil {
		return nil, err
	}
	outputName, outputRank, dim := pickDenseOutput(outs, opt.Output)
	if outputName == "" {
		return nil, fmt.Errorf("dense model %s has no usable tensor output", opt.ModelPath)
	}
	if opt.Dim <= 0 {
		opt.Dim = dim
	}
	if opt.Dim <= 0 {
		return nil, fmt.Errorf("dense model %s does not declare its width; set dense_dim", opt.ModelPath)
	}

	so, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer so.Destroy()

	sessions := opt.Sessions
	if sessions <= 0 {
		sessions = runtime.GOMAXPROCS(0)
		if sessions > 8 {
			sessions = 8
		}
	}
	if sessions < 1 {
		sessions = 1
	}
	if sessions > 1 {
		if err := so.SetIntraOpNumThreads(1); err != nil {
			return nil, fmt.Errorf("session threads: %w", err)
		}
	}

	names := make([]string, len(inputs))
	for i, in := range inputs {
		names[i] = in.name
	}
	pool := make(chan *ort.DynamicAdvancedSession, sessions)
	for i := 0; i < sessions; i++ {
		sess, err := ort.NewDynamicAdvancedSession(opt.ModelPath, names, []string{outputName}, so)
		if err != nil {
			close(pool)
			for s := range pool {
				_ = s.Destroy()
			}
			return nil, fmt.Errorf("create dense session %d/%d: %w", i+1, sessions, err)
		}
		pool <- sess
	}
	return &DenseONNX{opt: opt, pool: pool, inputs: inputs, outputName: outputName, outputRank: outputRank}, nil
}

// mapDenseInputs keeps the model's input order and classifies each input by
// name. Only the three BERT-style tensor inputs are supported, which is what a
// sentence-transformers export exposes.
func mapDenseInputs(info []ort.InputOutputInfo) ([]denseInput, error) {
	var out []denseInput
	for _, in := range info {
		n := strings.ToLower(in.Name)
		switch {
		case strings.Contains(n, "token_type"):
			out = append(out, denseInput{name: in.Name, kind: inputTokenTypes})
		case strings.Contains(n, "mask"):
			out = append(out, denseInput{name: in.Name, kind: inputMask})
		case strings.Contains(n, "ids"):
			out = append(out, denseInput{name: in.Name, kind: inputIDs})
		default:
			return nil, fmt.Errorf(
				"dense model input %q is not one of input_ids/attention_mask/token_type_ids; "+
					"export the model with `optimum-cli export onnx` or sentence-transformers", in.Name)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("dense model declares no inputs")
	}
	return out, nil
}

// pickDenseOutput chooses the graph output and the pooling width. A rank-2
// output is already pooled; a rank-3 one is token states and needs pooling.
func pickDenseOutput(info []ort.InputOutputInfo, want string) (name string, rank, dim int) {
	byName := make(map[string]ort.InputOutputInfo, len(info))
	for _, o := range info {
		byName[strings.ToLower(o.Name)] = o
	}
	try := func(n string) (string, int, int, bool) {
		o, ok := byName[strings.ToLower(n)]
		if !ok {
			return "", 0, 0, false
		}
		rank := len(o.Dimensions)
		var width int64
		if rank > 0 {
			width = o.Dimensions[rank-1]
		}
		return o.Name, rank, int(width), true
	}
	if want != "" {
		if n, r, d, ok := try(want); ok {
			return n, r, d
		}
		return "", 0, 0
	}
	for _, candidate := range []string{"sentence_embedding", "last_hidden_state", "pooler_output"} {
		if n, r, d, ok := try(candidate); ok {
			return n, r, d
		}
	}
	// Unknown names: prefer a 2-D output (already pooled), then a 3-D one.
	var bestName string
	var bestRank, bestDim int
	for _, o := range info {
		rank := len(o.Dimensions)
		if rank != 2 && rank != 3 {
			continue
		}
		var width int64
		if rank > 0 {
			width = o.Dimensions[rank-1]
		}
		if bestName == "" || (bestRank != 2 && rank == 2) {
			bestName, bestRank, bestDim = o.Name, rank, int(width)
		}
	}
	return bestName, bestRank, bestDim
}

// Dim implements Embedder.
func (d *DenseONNX) Dim() int { return d.opt.Dim }

// Multi implements Embedder.
func (d *DenseONNX) Multi() bool { return false }

// Sessions is how many forward passes may run at once (for diagnostics).
func (d *DenseONNX) Sessions() int { return cap(d.pool) }

// Name implements Embedder.
func (d *DenseONNX) Name() string { return "dense:" + filepath.Base(d.opt.ModelPath) }

// Close implements Embedder.
func (d *DenseONNX) Close() error {
	var first error
	for i := 0; i < cap(d.pool); i++ {
		if err := (<-d.pool).Destroy(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// EncodeDocs implements Embedder.
func (d *DenseONNX) EncodeDocs(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for start := 0; start < len(texts); start += d.opt.BatchSize {
		end := start + d.opt.BatchSize
		if end > len(texts) {
			end = len(texts)
		}
		if err := d.encodeBatch(texts[start:end], out[start:end]); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// EncodeQuery implements Embedder.
func (d *DenseONNX) EncodeQuery(text string) ([]float32, error) {
	out := make([][]float32, 1)
	if err := d.encodeBatch([]string{text}, out); err != nil {
		return nil, err
	}
	return out[0], nil
}

// encodeBatch pads to the batch's longest row. Unlike the int8 ColBERT export,
// a dense sentence-transformer is trained on padded batches and the mask is
// what both attention and pooling honour, so the result does not depend on who
// else is in the batch.
func (d *DenseONNX) encodeBatch(texts []string, out [][]float32) error {
	rows := make([][]int32, len(texts))
	wrap := d.opt.ClsID != 0 || d.opt.SepID != 0
	maxLen := 0
	for i, t := range texts {
		toks, err := native.Tokenize(d.opt.TokenizerPath, t)
		if err != nil {
			return err
		}
		limit := d.opt.MaxLen
		if wrap {
			limit -= 2
		}
		if limit > 0 && len(toks) > limit {
			toks = toks[:limit]
		}
		ids := make([]int32, 0, len(toks)+2)
		if wrap {
			ids = append(ids, d.opt.ClsID)
		}
		ids = append(ids, toks...)
		if wrap {
			ids = append(ids, d.opt.SepID)
		}
		if len(ids) == 0 {
			ids = []int32{0}
		}
		rows[i] = ids
		if len(ids) > maxLen {
			maxLen = len(ids)
		}
	}
	if maxLen == 0 {
		return nil
	}

	n := len(texts)
	ids := make([]int64, n*maxLen)
	mask := make([]int64, n*maxLen)
	for b := range rows {
		for j, id := range rows[b] {
			ids[b*maxLen+j] = int64(id)
			mask[b*maxLen+j] = 1
		}
	}

	sess := <-d.pool
	defer func() { d.pool <- sess }()

	shape := []int64{int64(n), int64(maxLen)}
	values := make([]ort.Value, 0, len(d.inputs))
	for _, in := range d.inputs {
		var data []int64
		switch in.kind {
		case inputMask:
			data = mask
		case inputTokenTypes:
			data = make([]int64, n*maxLen)
		default:
			data = ids
		}
		t, err := ort.NewTensor(ort.NewShape(shape...), data)
		if err != nil {
			for _, v := range values {
				v.Destroy()
			}
			return err
		}
		values = append(values, t)
	}
	defer func() {
		for _, v := range values {
			v.Destroy()
		}
	}()

	outShape := []int64{int64(n), int64(maxLen), int64(d.opt.Dim)}
	if d.outputRank == 2 {
		outShape = []int64{int64(n), int64(d.opt.Dim)}
	}
	outT, err := ort.NewEmptyTensor[float32](ort.NewShape(outShape...))
	if err != nil {
		return err
	}
	defer outT.Destroy()

	if err := sess.Run(values, []ort.Value{outT}); err != nil {
		return fmt.Errorf("dense session run: %w", err)
	}
	data := outT.GetData()
	dim := d.opt.Dim
	for b := range rows {
		vec := make([]float32, dim)
		if d.outputRank == 2 {
			base := b * dim
			if base+dim > len(data) {
				return fmt.Errorf("dense output is shorter than %dx%d", n, dim)
			}
			copy(vec, data[base:base+dim])
			Normalize(vec)
			out[b] = vec
			continue
		}
		nToks := len(rows[b])
		if d.opt.Pooling == "cls" {
			base := b * maxLen * dim
			if base+dim > len(data) {
				return fmt.Errorf("dense output is shorter than %dx%dx%d", n, maxLen, dim)
			}
			copy(vec, data[base:base+dim])
		} else {
			var count int
			for j := 0; j < nToks; j++ {
				off := (b*maxLen + j) * dim
				if off+dim > len(data) {
					break
				}
				for k := 0; k < dim; k++ {
					vec[k] += data[off+k]
				}
				count++
			}
			if count == 0 {
				return fmt.Errorf("dense text %d produced no tokens", b)
			}
			inv := float32(1) / float32(count)
			for k := range vec {
				vec[k] *= inv
			}
		}
		Normalize(vec)
		out[b] = vec
	}
	return nil
}
