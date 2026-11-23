package config

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// Config is the on-disk configuration. It is written on first run and can be
// edited by hand; `rgrep config` prints the resolved values.
type Config struct {
	// DataDir holds the sqlite catalog, logs and the daemon socket. The
	// catalog only remembers which files are indexed; it is not a search
	// index.
	DataDir string `json:"data_dir"`
	// ModelDir points at the ColBERT ONNX export (LateOn-Code-edge).
	ModelDir string `json:"model_dir"`
	// ORTLibDir contains libonnxruntime.*.dylib / .so / .dll.
	ORTLibDir string `json:"ort_lib_dir"`

	// Embedding selects the vector shape and therefore the engine index:
	//
	//   - "colbert" (default): one vector per token in a MULTIVECTOR column,
	//     recalled by PLAID with MaxSim. ModelDir is the encoder.
	//   - "dense": one pooled vector per unit in a VECTOR column, recalled by
	//     HNSW under DenseMetric. DenseModelDir is the encoder (ONNX + ORT).
	//   - "model2vec": the same VECTOR + HNSW shape, encoded by a static
	//     Model2Vec table instead of a transformer. Model2VecDir is the model
	//     directory (config.json + model.safetensors + tokenizer.json); no
	//     onnxruntime is loaded at all.
	//
	// All three are schema-level: the column type is part of the table, so
	// switching means a new infinity_table (or a fresh data dir) and a full
	// re-index.
	Embedding string `json:"embedding"`
	// DenseModelDir holds a sentence-transformers-style ONNX export plus its
	// tokenizer.json (model.onnx or model_int8.onnx).
	DenseModelDir string `json:"dense_model_dir"`
	// Model2VecDir is a Model2Vec model directory as written by
	// `StaticModel.save_pretrained`: config.json, model.safetensors and
	// tokenizer.json. The width comes from the table, not from DenseDim.
	Model2VecDir string `json:"model2vec_dir"`
	// DenseDim is the expected width. 0 reads it from the model's output shape;
	// set it when the export leaves the dimension dynamic.
	DenseDim int `json:"dense_dim"`
	// DenseMaxLen caps tokens per unit (512 is the usual limit).
	DenseMaxLen int `json:"dense_max_len"`
	// DensePooling is "mean" (default) or "cls", used when the model exposes
	// token states instead of a pooled sentence embedding.
	DensePooling string `json:"dense_pooling"`
	// DenseOutput names the graph output; empty auto-detects.
	DenseOutput string `json:"dense_output"`
	// DenseClsID / DenseSepID wrap the tokens (BERT family defaults 101/102).
	// Set both to 0 when the tokenizer already adds them.
	DenseClsID int32 `json:"dense_cls_id"`
	DenseSepID int32 `json:"dense_sep_id"`
	// DenseMetric is the HNSW distance: "cosine" (default), "l2" or "ip".
	DenseMetric string `json:"dense_metric"`
	// HNSWM / HNSWEfConstruction are the HNSW build parameters (0 = engine
	// defaults, 16 / 200).
	HNSWM              int `json:"hnsw_m"`
	HNSWEfConstruction int `json:"hnsw_ef_construction"`

	// InfinityMode is "embedded" (default) or "remote". Embedded runs the
	// engine inside the daemon, on a unix socket in the data dir, so there is
	// no server to install, no port to pick and nothing to supervise: the
	// engine's lifetime is the daemon's. Remote connects to a server someone
	// else runs, over TCP.
	InfinityMode string `json:"infinity_mode"`
	// InfinityAddr is the Thrift endpoint used in remote mode. PLAID
	// multi-vector recall, BM25 full-text recall and the fusion of the two all
	// run in the engine either way.
	InfinityAddr string `json:"infinity_addr"`
	// InfinityLibDir holds the embedded engine's shared library
	// (libinfinity_embedded.dylib / .so). Empty resolves to `<exe dir>/lib`.
	InfinityLibDir string `json:"infinity_lib_dir"`
	// InfinityDB / InfinityTable name the unit table inside Infinity.
	InfinityDB    string `json:"infinity_db"`
	InfinityTable string `json:"infinity_table"`
	// TensorTopN / TextTopN bound each recall branch; FusionTopN bounds the
	// fused candidate list rgrep re-ranks.
	TensorTopN int `json:"tensor_topn"`
	TextTopN   int `json:"text_topn"`
	FusionTopN int `json:"fusion_topn"`
	// Fusion is "weighted_sum" (default, honours --alpha) or "rrf".
	Fusion string `json:"fusion"`

	// Encoder parameters, mirrored from onnx_config.json.
	Dim      int   `json:"dim"`
	DocLen   int   `json:"doc_len"`
	QueryLen int   `json:"query_len"`
	ClsID    int32 `json:"cls_id"`
	SepID    int32 `json:"sep_id"`
	PrefixQ  int32 `json:"prefix_q"`
	PrefixD  int32 `json:"prefix_d"`

	// PlaidCentroids is the PLAID coarse-centroid budget, and the single
	// parameter that decides how long a full index pass takes. The engine sizes
	// it as `16*sqrt(token vectors)` by default - 32 768 for a 130 k-unit repo -
	// and its k-means costs `min(tokens, 256*k) * k * dims` per iteration, so the
	// default turns indexing into hours of training before the first query can
	// be answered (measured: a full ragflow pass stalled 8+ minutes inside
	// `PlaidIndex::train` at 48 k units, at 4.3 GB RSS). 1024 keeps a full build
	// in seconds; 0 restores the engine's own choice.
	PlaidCentroids int `json:"plaid_centroids"`

	// VecIndex picks the multi-vector recall index: "plaid" (default) keeps the
	// IVF + residual-code index, "smve" builds the sparse-projection index
	// (`SMVETransform` + BMP) whose candidates are re-ranked with the exact
	// MaxSim read back from the column. The two answer with different scores -
	// PLAID scores the dequantized codes, SMVE the stored float32 vectors - so
	// the setting is recorded here rather than inferred.
	VecIndex string `json:"vec_index"`
	// SMVEWidth / SMVETopK are the SMVE projection width (the sparse term space)
	// and the tokens each side keeps. Zero leaves the engine's defaults
	// (width 2048, topk 8).
	SMVEWidth int `json:"smve_width"`
	SMVETopK  int `json:"smve_topk"`

	// BatchSize is the ONNX batch size used while indexing.
	BatchSize int `json:"batch_size"`
	// Sessions is how many ONNX sessions run forward passes at once while
	// indexing. 0 means one per CPU; 1 keeps the single-session behaviour.
	Sessions int `json:"sessions"`
	// MaxFileBytes skips huge files (generated code, vendored blobs).
	MaxFileBytes int64 `json:"max_file_bytes"`
	// MaxEmbedChars truncates the structured text before tokenization.
	//
	// It has to cover `DocLen` tokens and no more: the encoder cuts the token
	// list to `DocLen-2` whatever arrives, so a budget of 8000 characters spent
	// ~87% of every tokenizer pass on tokens that were then thrown away. Code
	// runs about three characters per token, so 1600 characters is ~500 tokens -
	// comfortably over a 126-token document.
	MaxEmbedChars int `json:"max_embed_chars"`
	// IdleTimeout is how long the daemon lingers without work.
	IdleTimeout Duration `json:"idle_timeout"`
	// Ignore lists extra path patterns to skip (substring match on the relative path).
	Ignore []string `json:"ignore"`

	// ModelPath/TokenizerPath are derived (the ColBERT encoder).
	ModelPath     string `json:"-"`
	TokenizerPath string `json:"-"`
	// DenseModelPath/DenseTokenizerPath are the dense encoder's derived paths.
	DenseModelPath     string `json:"-"`
	DenseTokenizerPath string `json:"-"`
	DBPath             string `json:"-"`
	SocketPath         string `json:"-"`
	LockPath           string `json:"-"`
	// InfinityHost/InfinityPort are InfinityAddr split apart.
	InfinityHost string `json:"-"`
	InfinityPort int    `json:"-"`

	// Derived from InfinityMode: where the embedded engine keeps its files,
	// the socket it serves, and whether it is embedded at all.
	InfinityEmbedded bool   `json:"-"`
	InfinityDataDir  string `json:"-"`
	InfinitySocket   string `json:"-"`
}

// Duration is a time.Duration that marshals as a string ("30m").
type Duration time.Duration

func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(time.Duration(d).String())
}

func (d *Duration) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	v, err := time.ParseDuration(s)
	if err != nil {
		return err
	}
	*d = Duration(v)
	return nil
}

// DefaultDataDir is per-user application data.
func DefaultDataDir() string {
	base, err := os.UserConfigDir()
	if err != nil || base == "" {
		home, _ := os.UserHomeDir()
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "rgrep")
}

func defaults() *Config {
	cfg := &Config{
		DataDir:            DefaultDataDir(),
		InfinityMode:       "embedded",
		InfinityAddr:       "127.0.0.1:23817",
		InfinityDB:         "rgrep",
		InfinityTable:      "code_units",
		TensorTopN:         200,
		TextTopN:           200,
		FusionTopN:         200,
		Fusion:             "weighted_sum",
		Embedding:          "colbert",
		DenseMaxLen:        512,
		DensePooling:       "mean",
		DenseClsID:         101,
		DenseSepID:         102,
		DenseMetric:        "cosine",
		HNSWM:              16,
		HNSWEfConstruction: 200,
		Dim:                48,
		// DocLen is the per-unit token cap of the late-interaction encoding: a
		// unit costs `min(tokens, DocLen-2) * Dim * 4` bytes **twice** - once in
		// the row's MULTIVECTOR column, once in the PLAID chunk - so its price is
		// the dominant term in the index size. Measured on ragflow's
		// `internal/agent` (355 files, 5 772 units): 512 gave ~185 KB per unit,
		// 256 gave 46.6 KB, 128 gives 32 KB and encodes the same file in 1.9x
		// less time, with the same results on the queries that were compared.
		// 256 therefore costs ~2x the disk and ~2x the time of 128, and only
		// pays off for prose-heavy corpora whose meaning sits past the first
		// ~500 characters. Raise it there, not by default.
		DocLen:         128,
		QueryLen:       256,
		ClsID:          50281,
		SepID:          50282,
		PrefixQ:        50368,
		PrefixD:        50369,
		PlaidCentroids: 1024,
		VecIndex:       "plaid",
		SMVEWidth:      2048,
		SMVETopK:       8,
		BatchSize:      8,
		MaxFileBytes:   1 << 20,
		MaxEmbedChars:  1600,
		IdleTimeout:    Duration(30 * time.Minute),
		Ignore:         []string{},
	}
	// A local development fallback: the workspace checkout of the model.
	if home, err := os.UserHomeDir(); err == nil {
		candidate := filepath.Join(home, "codebase", "workspace", "LateOn-Code-edge")
		if st, err := os.Stat(filepath.Join(candidate, "model_int8.onnx")); err == nil && !st.IsDir() {
			cfg.ModelDir = candidate
		}
	}
	return cfg
}

// Load reads (or creates) the config file in the data dir.
func Load(dataDirOverride string) (*Config, error) {
	cfg := defaults()
	if v := os.Getenv("RGREP_HOME"); v != "" {
		cfg.DataDir = v
	}
	if dataDirOverride != "" {
		cfg.DataDir = dataDirOverride
	}
	path := filepath.Join(cfg.DataDir, "config.json")
	if b, err := os.ReadFile(path); err == nil {
		if err := json.Unmarshal(b, cfg); err != nil {
			return nil, fmt.Errorf("parse %s: %w", path, err)
		}
	} else if !os.IsNotExist(err) {
		return nil, err
	}

	if v := os.Getenv("RGREP_MODEL_DIR"); v != "" {
		cfg.ModelDir = v
	}
	if v := os.Getenv("RGREP_EMBEDDING"); v != "" {
		cfg.Embedding = v
	}
	if v := os.Getenv("RGREP_DENSE_MODEL_DIR"); v != "" {
		cfg.DenseModelDir = v
	}
	if v := os.Getenv("RGREP_MODEL2VEC_DIR"); v != "" {
		cfg.Model2VecDir = v
	}
	if v := os.Getenv("RGREP_INFINITY_MODE"); v != "" {
		cfg.InfinityMode = v
	}
	if v := os.Getenv("RGREP_INFINITY_ADDR"); v != "" {
		cfg.InfinityAddr = v
		// An explicit endpoint means "talk to that server", which is how the
		// tests and a shared deployment run rgrep.
		cfg.InfinityMode = "remote"
	}
	cfg.InfinityHost, cfg.InfinityPort = splitAddr(cfg.InfinityAddr)
	cfg.InfinityEmbedded = !strings.EqualFold(strings.TrimSpace(cfg.InfinityMode), "remote")
	cfg.InfinityDataDir = filepath.Join(cfg.DataDir, "infinity")
	cfg.InfinitySocket = socketPathNamed(cfg.DataDir, "infinity.sock", "rgrep-infinity-%d.sock")
	if cfg.ModelDir == "" {
		cfg.ModelDir = filepath.Join(cfg.DataDir, "models", "LateOn-Code-edge")
	}
	if cfg.Embedding == "" {
		cfg.Embedding = "colbert"
	}
	if cfg.DenseModelDir == "" {
		cfg.DenseModelDir = filepath.Join(cfg.DataDir, "models", "dense")
	}
	if cfg.Model2VecDir == "" {
		cfg.Model2VecDir = filepath.Join(cfg.DataDir, "models", "model2vec")
	}
	switch cfg.Mode() {
	case "colbert", "dense", "model2vec":
	default:
		return nil, fmt.Errorf(
			`unknown "embedding" %q; use "colbert" (PLAID), "dense" (ONNX + HNSW) or "model2vec" (Model2Vec + HNSW)`,
			cfg.Embedding)
	}
	if cfg.ORTLibDir == "" {
		if exe, err := os.Executable(); err == nil {
			cfg.ORTLibDir = filepath.Join(filepath.Dir(exe), "onnxruntime", "lib")
		}
	}
	if cfg.InfinityLibDir == "" {
		if exe, err := os.Executable(); err == nil {
			cfg.InfinityLibDir = filepath.Join(filepath.Dir(exe), "lib")
		}
	}

	// int8 first: it is 4x smaller and the accuracy delta on code retrieval is small.
	for _, name := range []string{"model_int8.onnx", "model.onnx"} {
		p := filepath.Join(cfg.ModelDir, name)
		if _, err := os.Stat(p); err == nil {
			cfg.ModelPath = p
			break
		}
	}
	cfg.TokenizerPath = filepath.Join(cfg.ModelDir, "tokenizer.json")
	// int8 first here too: a quantised dense export is ~4x smaller and code
	// retrieval loses little.
	for _, name := range []string{"model_int8.onnx", "model.onnx"} {
		p := filepath.Join(cfg.DenseModelDir, name)
		if _, err := os.Stat(p); err == nil {
			cfg.DenseModelPath = p
			break
		}
	}
	cfg.DenseTokenizerPath = filepath.Join(cfg.DenseModelDir, "tokenizer.json")
	cfg.DBPath = filepath.Join(cfg.DataDir, "rgrep.db")
	cfg.SocketPath = socketPath(cfg.DataDir)
	cfg.LockPath = filepath.Join(cfg.DataDir, "daemon.lock")
	return cfg, nil
}

// Dense reports whether the single-vector (HNSW) embedding is selected.
//
// Both "dense" (ONNX transformer) and "model2vec" (static table) produce one
// pooled vector per unit, so they share the VECTOR column and the HNSW index.
func (c *Config) Dense() bool {
	mode := c.Mode()
	return mode == "dense" || mode == "model2vec"
}

// Mode is the normalised embedding mode recorded in the catalog.
func (c *Config) Mode() string {
	mode := strings.ToLower(strings.TrimSpace(c.Embedding))
	if mode == "" {
		return "colbert"
	}
	return mode
}

// splitAddr turns "host:port" into its parts, defaulting to the Infinity
// Thrift port when either half is missing or unparsable.
func splitAddr(addr string) (string, int) {
	host, portStr, err := net.SplitHostPort(strings.TrimSpace(addr))
	if err != nil {
		return "127.0.0.1", 23817
	}
	port, err := strconv.Atoi(portStr)
	if err != nil || port <= 0 || port > 65535 {
		port = 23817
	}
	if host == "" {
		host = "127.0.0.1"
	}
	return host, port
}

// socketPath keeps the UDS path short (macOS caps it at 104 bytes).
func socketPath(dataDir string) string {
	return socketPathNamed(dataDir, "daemon.sock", "rgrep-%d.sock")
}

// socketPathNamed is [`socketPath`] for a socket with its own name, falling
// back to the temp dir when the data dir path would be too long.
func socketPathNamed(dataDir, name, fallbackFmt string) string {
	p := filepath.Join(dataDir, name)
	if len(p) <= 96 && runtime.GOOS != "windows" {
		return p
	}
	return filepath.Join(os.TempDir(), fmt.Sprintf(fallbackFmt, os.Getuid()))
}

// EnsureDirs creates the data directory layout and writes a default config on
// first run so users can find and edit it.
func (c *Config) EnsureDirs() error {
	for _, dir := range []string{c.DataDir, filepath.Join(c.DataDir, "logs")} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	path := filepath.Join(c.DataDir, "config.json")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		b, err := json.MarshalIndent(c, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(path, b, 0o644); err != nil {
			return err
		}
	}
	return nil
}
