package daemon

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"

	"github.com/infiniflow/rgrep/internal/app"
	"github.com/infiniflow/rgrep/internal/config"
	"github.com/infiniflow/rgrep/internal/index"
	"github.com/infiniflow/rgrep/internal/search"
)

type instance struct {
	PID      int    `json:"pid"`
	Hostname string `json:"hostname"`
	Socket   string `json:"socket"`
	Started  int64  `json:"started_unix"`
}

// Server is the rgrep daemon: the only process that writes the database.
type Server struct {
	cfg    *config.Config
	app    *app.App
	ln     net.Listener
	logger *log.Logger

	inflight   atomic.Int64
	lastActive atomic.Int64
	started    time.Time
	closed     atomic.Bool
}

// Serve runs the daemon until it is shut down or goes idle.
func Serve(cfg *config.Config, logw io.Writer) error {
	if logw == nil {
		logw = io.Discard
	}
	logger := log.New(logw, "", log.LstdFlags|log.Lmicroseconds)

	if inst, ok := readInstance(cfg.LockPath); ok && inst.Hostname == hostname() && processAlive(inst.PID) {
		// Somebody else already owns this data dir.
		if c, err := net.DialTimeout("unix", inst.Socket, 500*time.Millisecond); err == nil {
			c.Close()
			logger.Printf("already running pid=%d socket=%s", inst.PID, inst.Socket)
			return nil
		}
	}

	a, err := app.Open(cfg)
	if err != nil {
		return err
	}
	defer a.Close()

	_ = os.Remove(cfg.SocketPath)
	ln, err := net.Listen("unix", cfg.SocketPath)
	if err != nil {
		return fmt.Errorf("listen %s: %w", cfg.SocketPath, err)
	}
	defer ln.Close()
	_ = os.Chmod(cfg.SocketPath, 0o600)

	s := &Server{cfg: cfg, app: a, ln: ln, logger: logger, started: time.Now()}
	s.lastActive.Store(time.Now().UnixNano())

	writeInstance(cfg.LockPath, instance{
		PID: os.Getpid(), Hostname: hostname(), Socket: cfg.SocketPath, Started: time.Now().Unix(),
	})
	defer os.Remove(cfg.LockPath)
	defer os.Remove(cfg.SocketPath)

	go s.idleWatcher()

	logger.Printf("serving on %s (embedder=%s, pid=%d)", cfg.SocketPath, a.Note, os.Getpid())
	for {
		conn, err := ln.Accept()
		if err != nil {
			if s.closed.Load() {
				return nil
			}
			return err
		}
		go s.handle(conn)
	}
}

func (s *Server) idleWatcher() {
	idle := time.Duration(s.cfg.IdleTimeout)
	if idle <= 0 {
		return
	}
	tick := time.NewTicker(30 * time.Second)
	defer tick.Stop()
	for range tick.C {
		if s.closed.Load() {
			return
		}
		if s.inflight.Load() > 0 {
			continue
		}
		last := time.Unix(0, s.lastActive.Load())
		if time.Since(last) > idle {
			s.logger.Printf("idle for %s, exiting", time.Since(last).Round(time.Second))
			s.Shutdown()
			return
		}
	}
}

// Shutdown stops accepting connections and closes the listener.
func (s *Server) Shutdown() {
	if s.closed.Swap(true) {
		return
	}
	_ = s.ln.Close()
}

func (s *Server) handle(conn net.Conn) {
	defer conn.Close()
	s.inflight.Add(1)
	defer func() {
		s.inflight.Add(-1)
		s.lastActive.Store(time.Now().UnixNano())
	}()

	dec := json.NewDecoder(bufio.NewReader(conn))
	enc := json.NewEncoder(conn)

	for {
		var req Request
		if err := dec.Decode(&req); err != nil {
			return
		}
		s.lastActive.Store(time.Now().UnixNano())
		res := s.dispatch(&req)
		if err := enc.Encode(&res); err != nil {
			return
		}
		if req.Method == "shutdown" {
			s.Shutdown()
			return
		}
	}
}

func (s *Server) dispatch(req *Request) Response {
	resp := Response{ID: req.ID}
	wrap := func(v any, err error) Response {
		if err != nil {
			resp.Error = err.Error()
			return resp
		}
		b, mErr := json.Marshal(v)
		if mErr != nil {
			resp.Error = mErr.Error()
			return resp
		}
		resp.Result = b
		return resp
	}

	switch req.Method {
	case "ping":
		return wrap(map[string]any{"ok": true, "pid": os.Getpid(), "abi": 1}, nil)

	case "status":
		files, units, err := s.app.Cat.Stats()
		if err != nil {
			return wrap(nil, err)
		}
		return wrap(Status{
			Running: true, PID: os.Getpid(), Socket: s.cfg.SocketPath,
			Embedder: s.app.Note, Infinity: s.app.Eng.Addr(), Table: s.app.Eng.Table(),
			Files: files, Units: units,
			UptimeS: int64(time.Since(s.started).Seconds()),
		}, nil)

	case "index":
		var p IndexParams
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return wrap(nil, err)
		}
		if p.Path == "" {
			p.Path = "."
		}
		stats, err := s.app.Indexer().Run(index.Options{
			Root: p.Path, Ignore: s.cfg.Ignore, MaxFileBytes: s.cfg.MaxFileBytes,
			MaxEmbedChars: s.cfg.MaxEmbedChars, Force: p.Force,
			// The daemon has no stderr of its own beyond its log, so a file the
			// engine refused is recorded there; the count reaches the client in
			// the stats.
			OnError: func(path string, err error) {
				log.Printf("index: skipped %s: %v", path, err)
			},
		})
		return wrap(stats, err)

	case "search":
		var p SearchParams
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return wrap(nil, err)
		}
		res, err := s.app.Searcher().Search(p.Query, search.Options{
			K: p.K, Alpha: p.Alpha, PathPrefix: p.Path, Lang: p.Lang,
			Kind: p.Kind, NoTests: p.NoTests,
		})
		if err != nil {
			return wrap(nil, err)
		}
		if !p.WithCode {
			for i := range res {
				res[i].Code = ""
			}
		}
		return wrap(res, nil)

	case "get":
		var p GetParams
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return wrap(nil, err)
		}
		u, err := s.app.Eng.UnitAt(p.Path, p.Line)
		if err != nil {
			return wrap(nil, err)
		}
		return wrap(u, nil)

	case "stats":
		files, units, err := s.app.Cat.Stats()
		if err != nil {
			return wrap(nil, err)
		}
		return wrap(map[string]any{
			"files": files, "units": units, "embedder": s.app.Note,
			"infinity": s.app.Eng.Addr(), "table": s.app.Eng.Table(),
		}, nil)

	case "shutdown":
		return wrap(map[string]any{"ok": true}, nil)

	default:
		return wrap(nil, fmt.Errorf("unknown method %q", req.Method))
	}
}

func writeInstance(path string, inst instance) {
	b, _ := json.MarshalIndent(inst, "", "  ")
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err == nil {
		_ = os.Rename(tmp, path)
	}
}

func readInstance(path string) (instance, bool) {
	var inst instance
	b, err := os.ReadFile(path)
	if err != nil {
		return inst, false
	}
	if err := json.Unmarshal(b, &inst); err != nil {
		return inst, false
	}
	return inst, true
}

func hostname() string {
	h, err := os.Hostname()
	if err != nil {
		return "unknown"
	}
	return h
}

func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	// On unix FindProcess always succeeds; signal 0 is the liveness probe.
	return p.Signal(syscallSignalZero) == nil
}

// LogPath is where the spawned daemon writes its output.
func LogPath(cfg *config.Config) string {
	return filepath.Join(cfg.DataDir, "logs", "daemon.log")
}
