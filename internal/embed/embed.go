// Package embed turns text into ColBERT-style multi-vectors: one vector per
// token, which is what the PLAID / MaxSim family needs.
package embed

import (
	"hash/fnv"
	"math"
)

// Embedder produces per-token vectors for documents and queries.
type Embedder interface {
	// Dim is the embedding width (48 for LateOn-Code-edge).
	Dim() int
	// EncodeDocs returns, for each input text, a flat slice of n*Dim token vectors.
	EncodeDocs(texts []string) ([][]float32, error)
	// EncodeQuery returns a flat slice of n*Dim token vectors for one query.
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
type HashEmbedder struct {
	dim  int
	name string
}

// NewHash builds a fallback embedder.
func NewHash(dim int) *HashEmbedder {
	if dim <= 0 {
		dim = 64
	}
	return &HashEmbedder{dim: dim, name: "hash"}
}

func (h *HashEmbedder) Dim() int    { return h.dim }
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

// EncodeDocs implements Embedder.
func (h *HashEmbedder) EncodeDocs(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		words := splitIdentifiers(t)
		flat := make([]float32, 0, len(words)*h.dim)
		for _, w := range words {
			flat = append(flat, h.vector(w)...)
		}
		out[i] = flat
	}
	return out, nil
}

// EncodeQuery implements Embedder.
func (h *HashEmbedder) EncodeQuery(text string) ([]float32, error) {
	words := splitIdentifiers(text)
	flat := make([]float32, 0, len(words)*h.dim)
	for _, w := range words {
		flat = append(flat, h.vector(w)...)
	}
	return flat, nil
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
