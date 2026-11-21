// Package app wires configuration, the local catalog, the Infinity backend and
// the encoder together. Both the daemon and the in-process (--in-process) code
// path use it.
package app

import (
	"fmt"
	"os"

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
	emb, note := buildEmbedder(cfg)
	a := &App{Cfg: cfg, Cat: cat, Emb: emb, Note: note}

	eng, err := infinity.Open(infinity.Config{
		Host:       cfg.InfinityHost,
		Port:       cfg.InfinityPort,
		Database:   cfg.InfinityDB,
		Table:      cfg.InfinityTable,
		Dim:        emb.Dim(),
		TensorTopN: cfg.TensorTopN,
		TextTopN:   cfg.TextTopN,
		FusionTopN: cfg.FusionTopN,
		Fusion:     cfg.Fusion,
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
	_ = cat.SetMeta("dim", fmt.Sprintf("%d", emb.Dim()))
	return a, nil
}

func buildEmbedder(cfg *config.Config) (embed.Embedder, string) {
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
	})
	if err != nil {
		return embed.NewHash(64), "hash fallback: " + err.Error()
	}
	return o, "onnx: " + o.Name()
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
