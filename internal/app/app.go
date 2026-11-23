// Package app wires configuration, the local catalog, the Infinity backend and
// the encoder together. Both the daemon and the in-process (--in-process) code
// path use it.
package app

import (
	"fmt"
	"os"
	"strconv"

	"github.com/infiniflow/rgrep/internal/config"
	"github.com/infiniflow/rgrep/internal/embed"
	"github.com/infiniflow/rgrep/internal/index"
	"github.com/infiniflow/rgrep/internal/infinity"
	"github.com/infiniflow/rgrep/internal/search"
	"github.com/infiniflow/rgrep/internal/store"
)

// App is an open handle on the index.
type App struct {
	Cfg  *config.Config
	Cat  *store.Store
	Eng  *infinity.Backend
	Emb  embed.Embedder
	Note string // how the embedder was resolved, shown by `status`
}

// Open connects to Infinity, opens the catalog and builds the encoder.
func Open(cfg *config.Config) (*App, error) {
	if err := cfg.EnsureDirs(); err != nil {
		return nil, err
	}
	cat, err := store.Open(cfg.DBPath)
	if err != nil {
		return nil, err
	}
	// The vector column's type is fixed when the table is created, so an index
	// built in one mode cannot be opened in the other: refuse with a message
	// that names the way out instead of failing later inside an insert.
	mode := "colbert"
	if cfg.Dense() {
		mode = "dense"
	}
	if prev := cat.GetMeta("embedding_mode"); prev != "" && prev != mode {
		_ = cat.Close()
		return nil, fmt.Errorf(
			"the index in %s was built with embedding %q; set \"embedding\": %q to keep reading it, "+
				"or point infinity_table (or data_dir) at a fresh location and re-index for %q",
			cfg.DataDir, prev, prev, mode)
	}
	emb, note := buildEmbedder(cfg)
	if prev := cat.GetMeta("dim"); prev != "" && prev != strconv.Itoa(emb.Dim()) {
		_ = emb.Close()
		_ = cat.Close()
		return nil, fmt.Errorf(
			"the index in %s holds %s-dimension vectors, but the encoder produces %d (%s); "+
				"restore the previous model or re-index into a fresh infinity_table/data_dir",
			cfg.DataDir, prev, emb.Dim(), note)
	}
	a := &App{Cfg: cfg, Cat: cat, Emb: emb, Note: note}

	vectorType := infinity.VectorMulti
	if !emb.Multi() {
		vectorType = infinity.VectorDense
	}
	eng, err := infinity.Open(infinity.Config{
		Host:               cfg.InfinityHost,
		Port:               cfg.InfinityPort,
		Database:           cfg.InfinityDB,
		Table:              cfg.InfinityTable,
		Embedded:           cfg.InfinityEmbedded,
		DataDir:            cfg.InfinityDataDir,
		Socket:             cfg.InfinitySocket,
		LibDir:             cfg.InfinityLibDir,
		Dim:                emb.Dim(),
		VectorType:         vectorType,
		Metric:             cfg.DenseMetric,
		HNSWM:              cfg.HNSWM,
		HNSWEfConstruction: cfg.HNSWEfConstruction,
		PlaidCentroids:     cfg.PlaidCentroids,
		TensorTopN:         cfg.TensorTopN,
		TextTopN:           cfg.TextTopN,
		FusionTopN:         cfg.FusionTopN,
		Fusion:             cfg.Fusion,
	})
	if err != nil {
		_ = emb.Close()
		_ = cat.Close()
		return nil, err
	}
	a.Eng = eng

	if prev := cat.GetMeta("embedder"); prev != "" && prev != emb.Name() {
		a.Note = fmt.Sprintf("%s (index was built with %q; run index --force)", note, prev)
	}
	_ = cat.SetMeta("embedder", emb.Name())
	_ = cat.SetMeta("embedding_mode", mode)
	_ = cat.SetMeta("dim", fmt.Sprintf("%d", emb.Dim()))
	return a, nil
}

func buildEmbedder(cfg *config.Config) (embed.Embedder, string) {
	if cfg.Dense() {
		return buildDenseEmbedder(cfg)
	}
	if cfg.ModelPath == "" {
		return embed.NewHash(64), "hash fallback: no onnx model found"
	}
	if _, err := os.Stat(cfg.TokenizerPath); err != nil {
		return embed.NewHash(64), "hash fallback: tokenizer.json missing"
	}
	o, err := embed.NewONNX(embed.ONNXOptions{
		ModelPath:     cfg.ModelPath,
		TokenizerPath: cfg.TokenizerPath,
		ORTLibDir:     cfg.ORTLibDir,
		Dim:           cfg.Dim,
		DocLen:        cfg.DocLen,
		QueryLen:      cfg.QueryLen,
		ClsID:         cfg.ClsID,
		SepID:         cfg.SepID,
		PadID:         50283,
		PrefixQ:       cfg.PrefixQ,
		PrefixD:       cfg.PrefixD,
		BatchSize:     cfg.BatchSize,
		Sessions:      cfg.Sessions,
	})
	if err != nil {
		return embed.NewHash(64), "hash fallback: " + err.Error()
	}
	return o, "onnx: " + o.Name()
}

// buildDenseEmbedder opens the single-vector model. Without a usable export it
// falls back to the hash encoder so the pipeline still runs, but the note says
// so and vectors carry no meaning beyond token overlap.
func buildDenseEmbedder(cfg *config.Config) (embed.Embedder, string) {
	if cfg.DenseModelPath == "" {
		return embed.NewHashDense(64), "hash fallback: no dense onnx model found (set dense_model_dir)"
	}
	if _, err := os.Stat(cfg.DenseTokenizerPath); err != nil {
		return embed.NewHashDense(64), "hash fallback: dense tokenizer.json missing"
	}
	d, err := embed.NewDenseONNX(embed.DenseOptions{
		ModelPath:     cfg.DenseModelPath,
		TokenizerPath: cfg.DenseTokenizerPath,
		ORTLibDir:     cfg.ORTLibDir,
		Dim:           cfg.DenseDim,
		MaxLen:        cfg.DenseMaxLen,
		Pooling:       cfg.DensePooling,
		Output:        cfg.DenseOutput,
		ClsID:         cfg.DenseClsID,
		SepID:         cfg.DenseSepID,
		BatchSize:     cfg.BatchSize,
		Sessions:      cfg.Sessions,
	})
	if err != nil {
		return embed.NewHashDense(64), "hash fallback: " + err.Error()
	}
	return d, "onnx dense: " + d.Name()
}

// Close releases resources.
func (a *App) Close() error {
	if a.Emb != nil {
		_ = a.Emb.Close()
	}
	if a.Eng != nil {
		_ = a.Eng.Close()
	}
	if a.Cat != nil {
		return a.Cat.Close()
	}
	return nil
}

// Indexer returns an indexer for this app.
func (a *App) Indexer() *index.Indexer { return index.New(a.Cat, a.Eng, a.Emb) }

// Searcher returns a searcher for this app.
func (a *App) Searcher() *search.Searcher { return search.New(a.Eng, a.Emb) }
