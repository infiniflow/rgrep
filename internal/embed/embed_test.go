package embed

import (
	"math"
	"testing"

	ort "github.com/infiniflow/onnxruntime_go"
)

// The two shapes are what the backend keys the column type off, so the buffer
// a text produces has to match Multi().
func TestHashShapes(t *testing.T) {
	multi := NewHash(8)
	if !multi.Multi() {
		t.Fatal("NewHash must be the multi-vector shape")
	}
	docs, err := multi.EncodeDocs([]string{"parse config file"})
	if err != nil {
		t.Fatal(err)
	}
	if len(docs[0]) == 0 || len(docs[0])%8 != 0 {
		t.Fatalf("multi document = %d floats, want a multiple of 8", len(docs[0]))
	}
	if len(docs[0]) == 8 {
		t.Fatal("multi document collapsed to one token vector")
	}

	dense := NewHashDense(8)
	if dense.Multi() {
		t.Fatal("NewHashDense must be the dense shape")
	}
	docs, err = dense.EncodeDocs([]string{"parse config file", "x"})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range docs {
		if len(v) != 8 {
			t.Fatalf("dense document %d = %d floats, want 8", i, len(v))
		}
		var sum float64
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		if math.Abs(sum-1) > 1e-5 {
			t.Fatalf("dense document %d is not normalised: |v|^2 = %f", i, sum)
		}
	}

	qv, err := dense.EncodeQuery("parse config file")
	if err != nil {
		t.Fatal(err)
	}
	if len(qv) != 8 {
		t.Fatalf("dense query = %d floats, want 8", len(qv))
	}
}

// The output choice decides whether the encoder pools or reads a sentence
// embedding, and the width it reports is what the column is created with.
func TestPickDenseOutput(t *testing.T) {
	rank3 := []ort.InputOutputInfo{{
		Name:       "last_hidden_state",
		Dimensions: ort.Shape{int64(-1), int64(-1), int64(768)},
	}}
	name, rank, dim := pickDenseOutput(rank3, "")
	if name != "last_hidden_state" || rank != 3 || dim != 768 {
		t.Fatalf("auto pick = %q rank %d dim %d", name, rank, dim)
	}
	name, rank, dim = pickDenseOutput(rank3, "missing")
	if name != "" || rank != 0 || dim != 0 {
		t.Fatalf("explicit missing output = %q %d %d, want empty", name, rank, dim)
	}

	// A model that also emits a pooled embedding: prefer it, no pooling needed.
	both := append([]ort.InputOutputInfo{{
		Name:       "sentence_embedding",
		Dimensions: ort.Shape{int64(-1), int64(1024)},
	}}, rank3...)
	name, rank, dim = pickDenseOutput(both, "")
	if name != "sentence_embedding" || rank != 2 || dim != 1024 {
		t.Fatalf("pooled output = %q rank %d dim %d", name, rank, dim)
	}

	// Unknown names still work when the rank says what to do.
	odd := []ort.InputOutputInfo{{Name: "emb", Dimensions: ort.Shape{int64(-1), int64(384)}}}
	name, rank, dim = pickDenseOutput(odd, "")
	if name != "emb" || rank != 2 || dim != 384 {
		t.Fatalf("unknown output = %q rank %d dim %d", name, rank, dim)
	}
}

// Only BERT-style tensor inputs are filled; anything else is a clear error
// rather than a silently wrong tensor.
func TestMapDenseInputs(t *testing.T) {
	ins := []ort.InputOutputInfo{
		{Name: "input_ids"},
		{Name: "attention_mask"},
		{Name: "token_type_ids"},
	}
	got, err := mapDenseInputs(ins)
	if err != nil {
		t.Fatal(err)
	}
	want := []denseInputKind{inputIDs, inputMask, inputTokenTypes}
	for i := range want {
		if got[i].kind != want[i] {
			t.Fatalf("input %d = %+v, want kind %d", i, got[i], want[i])
		}
	}
	if _, err := mapDenseInputs([]ort.InputOutputInfo{{Name: "pixel_values"}}); err == nil {
		t.Fatal("an unknown input must be rejected")
	}
}
