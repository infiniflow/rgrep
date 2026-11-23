// Model2Vec static embeddings: the Rust side owns the table and the tokenizer,
// Go only passes strings and receives floats.
//
// This is deliberately a handle API rather than JSON: a 256xN float matrix per
// batch through JSON would cost more than the arithmetic it describes.

package native

/*
#include <stdlib.h>
#include "rgrep_native.h"
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"runtime"
	"unsafe"
)

// Model2VecInfo mirrors the JSON that rg_m2v_info reports.
type Model2VecInfo struct {
	Dim        int    `json:"dim"`
	Rows       int    `json:"rows"`
	Vocab      int    `json:"vocab"`
	Dtype      string `json:"dtype"`
	Normalize  bool   `json:"normalize"`
	MaxLength  *int   `json:"max_length"`
	HasWeights bool   `json:"has_weights"`
	HasMapping bool   `json:"has_mapping"`
}

// Model2Vec is one loaded static model. Close it when the index is done with it:
// the table is 30+ MB and the tokenizer holds the vocabulary.
type Model2Vec struct {
	handle *C.RgModelHandle
	dim    int
	info   Model2VecInfo
}

// LoadModel2Vec opens a directory written by Model2Vec's save_pretrained:
// config.json, model.safetensors and tokenizer.json.
func LoadModel2Vec(dir string) (*Model2Vec, error) {
	cDir := C.CString(dir)
	defer C.free(unsafe.Pointer(cDir))

	var cerr *C.char
	handle := C.rg_m2v_load(cDir, &cerr)
	if err := takeError(cerr); err != nil {
		return nil, fmt.Errorf("load model2vec %s: %w", dir, err)
	}
	if handle == nil {
		return nil, fmt.Errorf("load model2vec %s: native library returned no handle", dir)
	}

	m := &Model2Vec{handle: handle, dim: int(C.rg_m2v_dim(handle))}
	if m.dim <= 0 {
		m.Close()
		return nil, fmt.Errorf("load model2vec %s: model reports dimension %d", dir, m.dim)
	}
	if err := m.readInfo(); err != nil {
		m.Close()
		return nil, fmt.Errorf("load model2vec %s: %w", dir, err)
	}
	runtime.SetFinalizer(m, (*Model2Vec).Close)
	return m, nil
}

func (m *Model2Vec) readInfo() error {
	p := C.rg_m2v_info(m.handle)
	if p == nil {
		return fmt.Errorf("model info is unavailable")
	}
	defer C.rg_free(p)
	if err := json.Unmarshal([]byte(C.GoString(p)), &m.info); err != nil {
		return fmt.Errorf("model info is not JSON: %w", err)
	}
	return nil
}

// Dim is the embedding width; it comes from the loaded table, not from config.
func (m *Model2Vec) Dim() int { return m.dim }

// Info reports the table shape and the arithmetic flags the model was saved with.
func (m *Model2Vec) Info() Model2VecInfo { return m.info }

// Encode returns one L2-normalised vector per text, in input order.
func (m *Model2Vec) Encode(texts []string) ([][]float32, error) {
	if m == nil || m.handle == nil {
		return nil, fmt.Errorf("model2vec: model is closed")
	}
	if len(texts) == 0 {
		return nil, nil
	}

	// The C side takes NUL-terminated strings plus explicit lengths, so a text
	// containing NUL bytes is still embedded exactly as given.
	ptrs := make([]*C.char, len(texts))
	lens := make([]C.size_t, len(texts))
	for i, text := range texts {
		ptrs[i] = C.CString(text)
		lens[i] = C.size_t(len(text))
	}
	defer func() {
		for _, p := range ptrs {
			C.free(unsafe.Pointer(p))
		}
	}()

	flat := make([]float32, len(texts)*m.dim)
	var cerr *C.char
	rc := C.rg_m2v_encode(
		m.handle,
		(**C.char)(unsafe.Pointer(&ptrs[0])),
		(*C.size_t)(unsafe.Pointer(&lens[0])),
		C.size_t(len(texts)),
		(*C.float)(unsafe.Pointer(&flat[0])),
		&cerr,
	)
	if err := takeError(cerr); err != nil {
		return nil, err
	}
	if rc != 0 {
		return nil, fmt.Errorf("model2vec: encode failed with status %d", int(rc))
	}

	out := make([][]float32, len(texts))
	for i := range out {
		start := i * m.dim
		out[i] = flat[start : start+m.dim]
	}
	return out, nil
}

// Close releases the native handle. It is safe to call twice.
func (m *Model2Vec) Close() {
	if m == nil || m.handle == nil {
		return
	}
	C.rg_m2v_free(m.handle)
	m.handle = nil
	runtime.SetFinalizer(m, nil)
}
