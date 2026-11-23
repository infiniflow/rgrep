// Package native is the cgo binding to the in-repo Rust static library
// (native/rgrep_native). It exposes exactly two capabilities:
//
//   - Split: tree-sitter based code unit extraction (returns JSON)
//   - Tokenize: the model's own HF tokenizer, so we never reimplement BPE in Go
//
// Everything else (walking, hashing, embedding, indexing) lives in pure Go.
package native

/*
#cgo CFLAGS: -I${SRCDIR}/../../native/rgrep_native/include
#cgo LDFLAGS: -L${SRCDIR}/../../native/rgrep_native/target/release -lrgrep_native -lm

#include <stdlib.h>

char* rg_codesplit(const char* lang, const char* path, const char* content,
                   size_t content_len, char** err_out);
char* rg_tokenize(const char* tokenizer_path, const char* text, size_t text_len,
                  char** err_out);
char* rg_languages(void);
void  rg_free(char* ptr);
int   rg_abi_version(void);
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"unsafe"
)

// ABIExpected is the codesplit JSON schema version this package understands.
const ABIExpected = 1

// Unit is one searchable code unit produced by tree-sitter.
type Unit struct {
	Kind      string   `json:"kind"`
	Name      string   `json:"name"`
	Parent    string   `json:"parent"`
	Signature string   `json:"signature"`
	Doc       string   `json:"doc"`
	Calls     []string `json:"calls"`
	StartLine int      `json:"start_line"`
	EndLine   int      `json:"end_line"`
	StartByte int      `json:"start_byte"`
	EndByte   int      `json:"end_byte"`
}

// SplitResult is the per-file extraction result.
type SplitResult struct {
	Language  string `json:"language"`
	Degraded  bool   `json:"degraded"`
	Truncated bool   `json:"truncated"`
	Units     []Unit `json:"units"`
}

// ABI returns the native library ABI version.
func ABI() int { return int(C.rg_abi_version()) }

func takeError(errPtr *C.char) error {
	if errPtr == nil {
		return nil
	}
	defer C.rg_free(errPtr)
	return fmt.Errorf("%s", C.GoString(errPtr))
}

// SupportedLanguages lists the language identifiers the native splitter knows.
func SupportedLanguages() ([]string, error) {
	p := C.rg_languages()
	if p == nil {
		return nil, fmt.Errorf("rg_languages returned nil")
	}
	defer C.rg_free(p)
	var out []string
	if err := json.Unmarshal([]byte(C.GoString(p)), &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Split extracts code units from `content` using the tree-sitter grammar for `lang`.
func Split(lang, path string, content []byte) (*SplitResult, error) {
	if len(content) == 0 {
		return &SplitResult{Language: lang}, nil
	}
	cLang := C.CString(lang)
	defer C.free(unsafe.Pointer(cLang))
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var errPtr *C.char
	// The splitter copies what it needs during the call, so a Go pointer is fine here.
	res := C.rg_codesplit(cLang, cPath, (*C.char)(unsafe.Pointer(&content[0])),
		C.size_t(len(content)), &errPtr)
	if res == nil {
		if err := takeError(errPtr); err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("codesplit failed")
	}
	defer C.rg_free(res)

	var out SplitResult
	if err := json.Unmarshal([]byte(C.GoString(res)), &out); err != nil {
		return nil, fmt.Errorf("decode codesplit output: %w", err)
	}
	return &out, nil
}

// Tokenize runs the HF tokenizer at tokenizerPath over text and returns raw ids
// (no special tokens added).
func Tokenize(tokenizerPath, text string) ([]int32, error) {
	cPath := C.CString(tokenizerPath)
	defer C.free(unsafe.Pointer(cPath))

	buf := []byte(text)
	var ptr *C.char
	if len(buf) > 0 {
		ptr = (*C.char)(unsafe.Pointer(&buf[0]))
	} else {
		empty := C.CString("")
		defer C.free(unsafe.Pointer(empty))
		ptr = empty
	}

	var errPtr *C.char
	res := C.rg_tokenize(cPath, ptr, C.size_t(len(buf)), &errPtr)
	if res == nil {
		if err := takeError(errPtr); err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("tokenize failed")
	}
	defer C.rg_free(res)

	var payload struct {
		IDs []int32 `json:"ids"`
	}
	if err := json.Unmarshal([]byte(C.GoString(res)), &payload); err != nil {
		return nil, fmt.Errorf("decode tokenize output: %w", err)
	}
	return payload.IDs, nil
}
