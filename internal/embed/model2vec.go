package embed

import (
	"fmt"

	"github.com/infiniflow/rgrep/internal/native"
)

// Model2VecEmbedder wraps a static (Model2Vec) model directory.
//
// It is a dense encoder by construction: one pooled vector per text, no token
// matrix, so the index side picks the HNSW path. The arithmetic (token table,
// optional weights/mapping, mean, L2) lives in native/rgrep_native because it
// is the part that has to match the Python reference bit for bit.
type Model2VecEmbedder struct {
	model *native.Model2Vec
	name  string
}

// NewModel2Vec opens a directory containing config.json, model.safetensors and
// tokenizer.json, as written by `StaticModel.save_pretrained`.
func NewModel2Vec(dir string) (*Model2VecEmbedder, error) {
	if dir == "" {
		return nil, fmt.Errorf("model2vec: model directory is empty (set model2vec_dir)")
	}
	model, err := native.LoadModel2Vec(dir)
	if err != nil {
		return nil, err
	}
	info := model.Info()
	return &Model2VecEmbedder{
		model: model,
		name:  fmt.Sprintf("model2vec:%s:%dd", info.Dtype, info.Dim),
	}, nil
}

// Dim is the width of the loaded table.
func (e *Model2VecEmbedder) Dim() int { return e.model.Dim() }

// Multi is false: a static model has no per-token output.
func (e *Model2VecEmbedder) Multi() bool { return false }

// EncodeDocs embeds each text independently; the native side normalises.
func (e *Model2VecEmbedder) EncodeDocs(texts []string) ([][]float32, error) {
	return e.model.Encode(texts)
}

// EncodeQuery embeds one query. Static models have no query/document prefixes,
// so this is the same call as EncodeDocs.
func (e *Model2VecEmbedder) EncodeQuery(text string) ([]float32, error) {
	vectors, err := e.model.Encode([]string{text})
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, fmt.Errorf("model2vec: encoder returned no vector")
	}
	return vectors[0], nil
}

// Name identifies the encoder and its dtype for the index metadata.
func (e *Model2VecEmbedder) Name() string { return e.name }

// Info exposes the table shape for `rgrep status` and for diagnostics.
func (e *Model2VecEmbedder) Info() native.Model2VecInfo { return e.model.Info() }

// Close releases the table.
func (e *Model2VecEmbedder) Close() error {
	e.model.Close()
	return nil
}
