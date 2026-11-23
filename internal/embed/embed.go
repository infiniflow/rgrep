// Package embed turns text into vectors for Infinity.
//
// Two shapes are supported and they decide the index, not just the encoder:
//
//   - multi: one vector per token, like ColBERT. Infinity stores the row in a
//     MULTIVECTOR column behind a PLAID index and scores with MaxSim.
//   - dense: one pooled vector per text. Infinity stores the row in a
//     VECTOR column behind an HNSW index and scores with cosine / L2 / IP.
//
// Multi() is therefore the switch the rest of the program reads: the indexer
// writes the same flat buffer either way, but the backend has to know how many
// vectors one text produced and which recall operator to emit.
package embed

import (
	"hash/fnv"
	"math"
)

// Embedder produces vectors for documents and queries.
type Embedder interface {
	// Dim is the embedding width (48 for LateOn-Code-edge, 768/1024 for a
	// dense code model).
	Dim() int
	// Multi reports the shape of EncodeDocs/EncodeQuery: true for one vector
	// per token (ColBERT), false for one pooled vector per text.
	Multi() bool
	// EncodeDocs returns, for each input text, either a flat slice of n*Dim
	// token vectors (Multi) or a single Dim-sized vector.
	EncodeDocs(texts []string) ([][]float32, error)
	// EncodeQuery is EncodeDocs for one query.
	EncodeQuery(text string) ([]float32, error)
	// Name identifies the encoder for the index metadata.
	Name() string
	Close() error
}

// Normalize scales v to unit length in place.
func Normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}

func normalizeRows(v []float32, dim int) {
	if dim <= 0 {
		return
	}
	for i := 0; i+dim <= len(v); i += dim {
		Normalize(v[i : i+dim])
	}
}

// HashEmbedder is a dependency-free fallback used when onnxruntime is not
// installed. It is a hashed bag-of-token embedding: useful for smoke tests and
// for exercising the whole pipeline, not for production relevance.
//
// Both shapes are produced by the same code: multi emits one vector per
// identifier token, dense averages them into one vector and normalises it.
type HashEmbedder struct {
	dim   int
	name  string
	multi bool
}

// NewHash builds a fallback embedder that produces per-token vectors.
func NewHash(dim int) *HashEmbedder {
	if dim <= 0 {
		dim = 64
	}
	return &HashEmbedder{dim: dim, name: "hash", multi: true}
}

// NewHashDense builds a fallback embedder that produces one vector per text.
func NewHashDense(dim int) *HashEmbedder {
	if dim <= 0 {
		dim = 64
	}
	return &HashEmbedder{dim: dim, name: "hash-dense", multi: false}
}

func (h *HashEmbedder) Dim() int     { return h.dim }
func (h *HashEmbedder) Multi() bool  { return h.multi }
func (h *HashEmbedder) Name() string { return h.name }
func (h *HashEmbedder) Close() error { return nil }

func (h *HashEmbedder) vector(token string) []float32 {
	v := make([]float32, h.dim)
	// Hash the token plus a few character trigrams so sub-word overlap helps.
	tokens := []string{token}
	if len(token) > 3 {
		for i := 0; i+3 <= len(token); i++ {
			tokens = append(tokens, token[i:i+3])
		}
	}
	for _, t := range tokens {
		f := fnv.New64a()
		f.Write([]byte(t))
		seed := f.Sum64()
		for d := 0; d < h.dim; d++ {
			seed = seed*6364136223846793005 + 1442695040888963407
			bit := (seed >> 33) & 1
			sign := float32(1)
			if bit == 0 {
				sign = -1
			}
			v[d] += sign
		}
	}
	Normalize(v)
	return v
}

// encode is one text in this embedder's shape.
func (h *HashEmbedder) encode(text string) []float32 {
	words := splitIdentifiers(text)
	if !h.multi {
		pooled := make([]float32, h.dim)
		for _, w := range words {
			v := h.vector(w)
			for d := range pooled {
				pooled[d] += v[d]
			}
		}
		Normalize(pooled)
		return pooled
	}
	flat := make([]float32, 0, len(words)*h.dim)
	for _, w := range words {
		flat = append(flat, h.vector(w)...)
	}
	return flat
}

// EncodeDocs implements Embedder.
func (h *HashEmbedder) EncodeDocs(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = h.encode(t)
	}
	return out, nil
}

// EncodeQuery implements Embedder.
func (h *HashEmbedder) EncodeQuery(text string) ([]float32, error) {
	return h.encode(text), nil
}

func splitIdentifiers(s string) []string {
	var out []string
	start := -1
	for i, r := range s {
		ok := r == '_' || r == '$' || r == '.' ||
			(r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')
		if ok && start < 0 {
			start = i
		}
		if !ok && start >= 0 {
			out = append(out, s[start:i])
			start = -1
		}
	}
	if start >= 0 {
		out = append(out, s[start:])
	}
	if len(out) > 512 {
		out = out[:512]
	}
	return out
}
