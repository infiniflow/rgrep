// Package cli implements the rgrep command line surface. Every subcommand talks
// to the daemon unless --in-process is given.
package cli

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/infiniflow/rgrep/internal/app"
	"github.com/infiniflow/rgrep/internal/config"
	"github.com/infiniflow/rgrep/internal/daemon"
	"github.com/infiniflow/rgrep/internal/index"
	"github.com/infiniflow/rgrep/internal/search"
	"github.com/infiniflow/rgrep/internal/store"
)

// Version is stamped by the build.
const Version = "0.1.0"

// Run dispatches the command line.
func Run(args []string) int {
	// --home is global and may appear anywhere.
	home := ""
	cleaned := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--home" && i+1 < len(args) {
			home = args[i+1]
			i++
			continue
		}
		if v, ok := strings.CutPrefix(a, "--home="); ok {
			home = v
			continue
		}
		cleaned = append(cleaned, a)
	}
	args = cleaned
	if len(args) == 0 {
		usage()
		return 2
	}
	cmd := args[0]
	rest := args[1:]

	switch cmd {
	case "-h", "--help", "help":
		usage()
		return 0
	case "version", "--version":
		fmt.Printf("rgrep %s (native abi %d)\n", Version, nativeABI())
		return 0
	}

	cfg, err := config.Load(home)
	if err != nil {
		fmt.Fprintln(os.Stderr, "config:", err)
		return 1
	}

	switch cmd {
	case "search":
		return cmdSearch(cfg, rest)
	case "index", "init":
		return cmdIndex(cfg, rest, cmd == "init")
	case "get":
		return cmdGet(cfg, rest)
	case "status":
		return cmdStatus(cfg, rest)
	case "stats":
		return cmdStats(cfg, rest)
	case "serve":
		return cmdServe(cfg, rest)
	case "shutdown":
		return cmdShutdown(cfg, rest)
	case "config":
		return cmdConfig(cfg, rest)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", cmd)
		usage()
		return 2
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `rgrep — local code search for agents

  rgrep index [path] [-y] [--force]      build/update the index for a repo
  rgrep search <query> [-k 10] [--json]  hybrid semantic + keyword search
  rgrep get <path>:<line>                show the unit enclosing a line
  rgrep status [--json]                  daemon + index status
  rgrep stats [--json]                   index counts
  rgrep config [--json]                  resolved configuration
  rgrep serve                            run the daemon in the foreground
  rgrep shutdown                         stop the daemon
  rgrep version

Global: --home <dir> to use an alternate data dir, --in-process to skip the daemon.
`)
}

func nativeABI() int {
	// Kept behind a tiny indirection so `version` works even if cgo init is slow.
	return abiValue
}

// splitArgs separates flags (and their values) from positionals so that
// `rgrep search "foo" -k 5` works the way people expect.
func splitArgs(args []string, valued map[string]bool) (flags, pos []string) {
	for i := 0; i < len(args); i++ {
		a := args[i]
		if strings.HasPrefix(a, "-") && a != "-" {
			if eq := strings.IndexByte(a, '='); eq > 0 {
				flags = append(flags, a)
				continue
			}
			flags = append(flags, a)
			name := a
			if strings.HasPrefix(name, "--") {
				name = "-" + name[1:]
			}
			if valued[name] && i+1 < len(args) {
				flags = append(flags, args[i+1])
				i++
			}
			continue
		}
		pos = append(pos, a)
	}
	return
}

func findHome(args []string) string {
	for i, a := range args {
		if a == "--home" && i+1 < len(args) {
			return args[i+1]
		}
		if v, ok := strings.CutPrefix(a, "--home="); ok {
			return v
		}
	}
	return ""
}

// stripHome removes the global --home flag (and its value) from the args
// handed to a subcommand.
func stripHome(args []string) []string {
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--home" {
			i++
			continue
		}
		if strings.HasPrefix(a, "--home=") {
			continue
		}
		out = append(out, a)
	}
	return out
}

var valuedFlags = map[string]bool{
	"-k": true, "-n": true, "--k": true, "--alpha": true, "--path": true,
	"--lang": true, "--kind": true, "--home": true, "--limit": true,
}

// --- search ---------------------------------------------------------------

func cmdSearch(cfg *config.Config, args []string) int {
	flags, pos := splitArgs(args, valuedFlags)
	fs := flag.NewFlagSet("search", flag.ContinueOnError)
	k := fs.Int("k", 10, "number of results")
	alpha := fs.Float64("alpha", 0.6, "semantic weight in the fusion (0..1)")
	pathPrefix := fs.String("path", "", "restrict to a path prefix")
	lang := fs.String("lang", "", "restrict to a language")
	kind := fs.String("kind", "", "restrict to a unit kind")
	noTests := fs.Bool("no-tests", false, "skip test files")
	withCode := fs.Bool("code", false, "include code snippets")
	asJSON := fs.Bool("json", false, "emit JSON")
	inProc := fs.Bool("in-process", false, "run without the daemon")
	fs.Bool("y", false, "accepted for scripting symmetry")
	if err := fs.Parse(flags); err != nil {
		return 2
	}
	query := strings.Join(pos, " ")
	if strings.TrimSpace(query) == "" {
		fmt.Fprintln(os.Stderr, "missing query")
		return 2
	}

	var results []search.Result
	if *inProc {
		a, err := app.Open(cfg)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		defer a.Close()
		results, err = a.Searcher().Search(query, search.Options{
			K: *k, Alpha: *alpha, PathPrefix: *pathPrefix, Lang: *lang,
			Kind: *kind, NoTests: *noTests,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		if !*withCode {
			for i := range results {
				results[i].Code = ""
			}
		}
	} else {
		err := daemon.Call(cfg, "search", daemon.SearchParams{
			Query: query, K: *k, Alpha: *alpha, Path: *pathPrefix, Lang: *lang,
			Kind: *kind, NoTests: *noTests, WithCode: *withCode,
		}, &results)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(results)
		return 0
	}
	if len(results) == 0 {
		fmt.Fprintln(os.Stderr, "no results")
		return 4
	}
	for _, r := range results {
		name := r.Name
		if r.Parent != "" {
			name = r.Parent + "." + r.Name
		}
		if name == "" {
			name = r.Kind
		}
		fmt.Printf("%s:%d-%d  %s  (%.3f)\n", r.Path, r.StartLn, r.EndLn, name, r.Score)
		if *withCode && r.Code != "" {
			for _, line := range strings.Split(truncateCode(r.Code, 12), "\n") {
				fmt.Printf("    %s\n", line)
			}
		}
	}
	return 0
}

func truncateCode(code string, maxLines int) string {
	lines := strings.Split(code, "\n")
	if len(lines) > maxLines {
		lines = append(lines[:maxLines], "…")
	}
	return strings.Join(lines, "\n")
}

// --- index ----------------------------------------------------------------

func cmdIndex(cfg *config.Config, args []string, isInit bool) int {
	flags, pos := splitArgs(args, valuedFlags)
	fs := flag.NewFlagSet("index", flag.ContinueOnError)
	force := fs.Bool("force", false, "re-extract every file")
	asJSON := fs.Bool("json", false, "emit JSON stats")
	inProc := fs.Bool("in-process", false, "run without the daemon")
	quiet := fs.Bool("quiet", false, "no progress output")
	fs.Bool("y", false, "assume yes (non-interactive)")
	if err := fs.Parse(flags); err != nil {
		return 2
	}
	root := "."
	if len(pos) > 0 {
		root = pos[0]
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	if st, err := os.Stat(abs); err != nil || !st.IsDir() {
		fmt.Fprintf(os.Stderr, "%s is not a directory\n", abs)
		return 1
	}

	var stats *index.Stats
	if *inProc {
		a, err := app.Open(cfg)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		defer a.Close()
		stats, err = a.Indexer().Run(index.Options{
			Root: abs, Ignore: cfg.Ignore, MaxFileBytes: cfg.MaxFileBytes,
			MaxEmbedChars: cfg.MaxEmbedChars, Force: *force,
			OnProgress: func(done, total int, path string) {
				if *quiet || *asJSON {
					return
				}
				if done%20 == 0 || done == total {
					fmt.Fprintf(os.Stderr, "  [%d/%d] %s\n", done, total, path)
				}
			},
			OnError: func(path string, err error) {
				fmt.Fprintf(os.Stderr, "  skipped %s: %v\n", path, err)
			},
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
	} else {
		err := daemon.Call(cfg, "index", daemon.IndexParams{Path: abs, Force: *force}, &stats)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
	}

	if stats == nil {
		fmt.Fprintln(os.Stderr, "index returned no stats")
		return 1
	}
	if *asJSON {
		b, _ := json.MarshalIndent(stats, "", "  ")
		fmt.Println(string(b))
		return 0
	}
	fmt.Printf("indexed %s: %d files scanned (%d added, %d changed, %d deleted, %d unchanged), %d units, %d failed, %s\n",
		abs, stats.Scanned, stats.Added, stats.Changed, stats.Deleted, stats.Unchanged,
		stats.Units, stats.Failed, stats.Duration.Round(time.Millisecond))
	return 0
}

// --- get ------------------------------------------------------------------

func cmdGet(cfg *config.Config, args []string) int {
	flags, pos := splitArgs(args, valuedFlags)
	fs := flag.NewFlagSet("get", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "emit JSON")
	inProc := fs.Bool("in-process", false, "run without the daemon")
	if err := fs.Parse(flags); err != nil {
		return 2
	}
	if len(pos) == 0 {
		fmt.Fprintln(os.Stderr, "usage: rgrep get <path>:<line>")
		return 2
	}
	path, line := splitPathLine(pos[0])
	var u *store.Unit
	if *inProc {
		a, err := app.Open(cfg)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		defer a.Close()
		if u, err = a.Eng.UnitAt(path, line); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
	} else {
		var got store.Unit
		if err := daemon.Call(cfg, "get", daemon.GetParams{Path: path, Line: line}, &got); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		u = &got
	}
	if *asJSON {
		b, _ := json.MarshalIndent(u, "", "  ")
		fmt.Println(string(b))
		return 0
	}
	fmt.Printf("%s:%d-%d  %s %s\n%s\n", u.Path, u.StartLine, u.EndLine, u.Kind, u.Name, u.Code)
	return 0
}

func splitPathLine(s string) (string, int) {
	i := strings.LastIndexByte(s, ':')
	if i < 0 {
		return s, 0
	}
	n, err := strconv.Atoi(s[i+1:])
	if err != nil {
		return s, 0
	}
	return s[:i], n
}

// --- status / stats / config ----------------------------------------------

func cmdStatus(cfg *config.Config, args []string) int {
	flags, _ := splitArgs(args, valuedFlags)
	fs := flag.NewFlagSet("status", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(flags); err != nil {
		return 2
	}
	var st daemon.Status
	if err := daemon.Call(cfg, "status", nil, &st); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	if *asJSON {
		b, _ := json.MarshalIndent(st, "", "  ")
		fmt.Println(string(b))
		return 0
	}
	fmt.Printf("pid      %d\nsocket   %s\nembedder %s\ninfinity %s (%s)\nfiles    %d\nunits    %d\nuptime   %ds\n",
		st.PID, st.Socket, st.Embedder, st.Infinity, st.Table, st.Files, st.Units, st.UptimeS)
	return 0
}

func cmdStats(cfg *config.Config, args []string) int {
	flags, _ := splitArgs(args, valuedFlags)
	fs := flag.NewFlagSet("stats", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(flags); err != nil {
		return 2
	}
	var out map[string]any
	if err := daemon.Call(cfg, "stats", nil, &out); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	if *asJSON {
		b, _ := json.MarshalIndent(out, "", "  ")
		fmt.Println(string(b))
		return 0
	}
	fmt.Printf("files %v  units %v  embedder %v  infinity %v (%v)\n",
		out["files"], out["units"], out["embedder"], out["infinity"], out["table"])
	return 0
}

func cmdConfig(cfg *config.Config, args []string) int {
	flags, _ := splitArgs(args, valuedFlags)
	fs := flag.NewFlagSet("config", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(flags); err != nil {
		return 2
	}
	if *asJSON {
		b, _ := json.MarshalIndent(cfg, "", "  ")
		fmt.Println(string(b))
		return 0
	}
	fmt.Printf("data_dir       %s\nmodel_dir      %s\nmodel          %s\nort_lib_dir    %s\ncatalog        %s\nsocket         %s\ninfinity       %s (%s.%s, fusion %s)\nembed dim      %d (doc_len %d, query_len %d)\n",
		cfg.DataDir, cfg.ModelDir, cfg.ModelPath, cfg.ORTLibDir, cfg.DBPath, cfg.SocketPath,
		cfg.InfinityAddr, cfg.InfinityDB, cfg.InfinityTable, cfg.Fusion,
		cfg.Dim, cfg.DocLen, cfg.QueryLen)
	return 0
}

// --- serve / shutdown -----------------------------------------------------

func cmdServe(cfg *config.Config, args []string) int {
	flags, _ := splitArgs(args, valuedFlags)
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	if err := fs.Parse(flags); err != nil {
		return 2
	}
	if err := cfg.EnsureDirs(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	return serve(cfg)
}

func cmdShutdown(cfg *config.Config, args []string) int {
	flags, _ := splitArgs(args, valuedFlags)
	fs := flag.NewFlagSet("shutdown", flag.ContinueOnError)
	if err := fs.Parse(flags); err != nil {
		return 2
	}
	var out map[string]any
	if err := daemon.Call(cfg, "shutdown", nil, &out); err != nil {
		// Already down is a success for scripting.
		fmt.Println("daemon not running")
		return 0
	}
	fmt.Println("daemon stopped")
	return 0
}
