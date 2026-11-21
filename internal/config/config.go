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

	// BatchSize is the ONNX batch size used while indexing.
	BatchSize int `json:"batch_size"`
	// MaxFileBytes skips huge files (generated code, vendored blobs).
	MaxFileBytes int64 `json:"max_file_bytes"`
	// MaxEmbedChars truncates the structured text before tokenization.
	MaxEmbedChars int `json:"max_embed_chars"`
	// IdleTimeout is how long the daemon lingers without work.
	IdleTimeout Duration `json:"idle_timeout"`
	// Ignore lists extra path patterns to skip (substring match on the relative path).
	Ignore []string `json:"ignore"`

	// ModelPath/TokenizerPath are derived.
	ModelPath     string `json:"-"`
	TokenizerPath string `json:"-"`
	DBPath        string `json:"-"`
	SocketPath    string `json:"-"`
	LockPath      string `json:"-"`
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
		DataDir:       DefaultDataDir(),
		InfinityMode:  "embedded",
		InfinityAddr:  "127.0.0.1:23817",
		InfinityDB:    "rgrep",
		InfinityTable: "code_units",
		TensorTopN:    200,
		TextTopN:      200,
		FusionTopN:    200,
		Fusion:        "weighted_sum",
		Dim:           48,
		DocLen:        512,
		QueryLen:      256,
		ClsID:         50281,
		SepID:         50282,
		PrefixQ:       50368,
		PrefixD:       50369,
		BatchSize:     8,
		MaxFileBytes:  1 << 20,
		MaxEmbedChars: 8000,
		IdleTimeout:   Duration(30 * time.Minute),
		Ignore:        []string{},
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
	cfg.DBPath = filepath.Join(cfg.DataDir, "rgrep.db")
	cfg.SocketPath = socketPath(cfg.DataDir)
	cfg.LockPath = filepath.Join(cfg.DataDir, "daemon.lock")
	return cfg, nil
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
