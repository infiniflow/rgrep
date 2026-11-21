// Package index walks a repository, extracts code units, embeds them and writes
// them to the store. It owns the incremental (file-state) bookkeeping.
package index

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/infiniflow/rgrep/internal/embed"
	"github.com/infiniflow/rgrep/internal/native"
	"github.com/infiniflow/rgrep/internal/store"
)

// Options controls one indexing pass.
type Options struct {
	Root          string
	Ignore        []string
	MaxFileBytes  int64
	MaxEmbedChars int
	// Force ignores the file-state cache and re-extracts everything.
	Force bool
	// OnProgress is called between files; may be nil.
	OnProgress func(done, total int, path string)
}

// Stats summarises a pass.
type Stats struct {
	Scanned   int           `json:"scanned"`
	Added     int           `json:"added"`
	Changed   int           `json:"changed"`
	Deleted   int           `json:"deleted"`
	Unchanged int           `json:"unchanged"`
	Units     int           `json:"units"`
	Failed    int           `json:"failed"`
	Duration  time.Duration `json:"duration_ms"`
}

var defaultSkipDirs = map[string]bool{
	".git": true, ".hg": true, ".svn": true, "node_modules": true, "vendor": true,
	"target": true, "dist": true, "build": true, "out": true, "bin": true,
	".venv": true, "venv": true, "__pycache__": true, ".mypy_cache": true,
	".pytest_cache": true, ".idea": true, ".vscode": true, ".next": true,
	".gradle": true, ".cargo": true, "Pods": true, "DerivedData": true,
	".rgrep": true, ".zvec-grep": true, ".colgrep": true,
}

var langByExt = map[string]string{
	".go": "go", ".py": "python", ".pyi": "python", ".rs": "rust",
	".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
	".ts": "typescript", ".mts": "typescript", ".cts": "typescript", ".tsx": "tsx",
}

// LanguageFor returns the tree-sitter language for a path, or "".
func LanguageFor(path string) string {
	return langByExt[strings.ToLower(filepath.Ext(path))]
}

// Engine is the document side: units go here, not into the catalog.
type Engine interface {
	ReplaceUnits(path string, units []store.Unit) error
	DeletePath(path string) error
}

// Indexer binds the catalog (incremental state), the engine (documents) and
// the embedder (token vectors).
type Indexer struct {
	cat store.Catalog
	eng Engine
	emb embed.Embedder
}

// New builds an indexer.
func New(cat store.Catalog, eng Engine, emb embed.Embedder) *Indexer {
	return &Indexer{cat: cat, eng: eng, emb: emb}
}

type candFile struct {
	rel     string
	abs     string
	lang    string
	size    int64
	mtimeNs int64
	hash    string
	changed bool
	added   bool
}

// Run performs one incremental indexing pass over opt.Root.
func (ix *Indexer) Run(opt Options) (*Stats, error) {
	start := time.Now()
	if opt.MaxFileBytes <= 0 {
		opt.MaxFileBytes = 1 << 20
	}
	if opt.MaxEmbedChars <= 0 {
		opt.MaxEmbedChars = 8000
	}
	root, err := filepath.Abs(opt.Root)
	if err != nil {
		return nil, err
	}

	known, err := ix.cat.Files()
	if err != nil {
		return nil, err
	}

	stats := &Stats{}
	var cands []candFile
	seen := make(map[string]bool, len(known))

	err = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // unreadable entry: skip, do not abort the pass
		}
		rel, relErr := filepath.Rel(root, p)
		if relErr != nil {
			return nil
		}
		rel = filepath.ToSlash(rel)
		if d.IsDir() {
			if rel == "." {
				return nil
			}
			if defaultSkipDirs[d.Name()] {
				return filepath.SkipDir
			}
			for _, pat := range opt.Ignore {
				if pat != "" && strings.Contains(rel, pat) {
					return filepath.SkipDir
				}
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		lang := LanguageFor(p)
		if lang == "" {
			return nil
		}
		for _, pat := range opt.Ignore {
			if pat != "" && strings.Contains(rel, pat) {
				return nil
			}
		}
		info, err := d.Info()
		if err != nil || info.Size() > opt.MaxFileBytes {
			return nil
		}
		seen[rel] = true
		stats.Scanned++
		cands = append(cands, candFile{
			rel: rel, abs: p, lang: lang, size: info.Size(), mtimeNs: info.ModTime().UnixNano(),
		})
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].rel < cands[j].rel })

	// Deletions first: cheap, and keeps the index consistent if we get killed.
	for rel := range known {
		if !seen[rel] {
			if err := ix.eng.DeletePath(rel); err != nil {
				return nil, fmt.Errorf("delete %s: %w", rel, err)
			}
			if err := ix.cat.DeleteFile(rel); err != nil {
				return nil, fmt.Errorf("delete %s: %w", rel, err)
			}
			stats.Deleted++
		}
	}

	// Hash pass (needs the content anyway, so read once per changed file).
	var work []candFile
	for i := range cands {
		c := &cands[i]
		prev, ok := known[c.rel]
		if ok && !opt.Force && prev.Size == c.size && prev.MtimeNs == c.mtimeNs {
			stats.Unchanged++
			continue
		}
		content, err := os.ReadFile(c.abs)
		if err != nil {
			stats.Failed++
			continue
		}
		sum := sha256.Sum256(content)
		c.hash = hex.EncodeToString(sum[:])
		if ok && !opt.Force && prev.Hash == c.hash {
			// Only the stat drifted (touch / branch switch): refresh metadata.
			stats.Unchanged++
			work = append(work, candFile{rel: c.rel, abs: c.abs, lang: c.lang, size: c.size,
				mtimeNs: c.mtimeNs, hash: c.hash, changed: false})
			continue
		}
		c.changed = ok
		c.added = !ok
		work = append(work, *c)
	}

	// Refresh stat-only entries in bulk (no re-extraction).
	for _, c := range work {
		if !c.changed && !c.added {
			if prev, ok := known[c.rel]; ok {
				prev.MtimeNs = c.mtimeNs
				prev.Size = c.size
				if prev.Hash == "" {
					prev.Hash = c.hash
				}
				// Re-upserting the file row keeps mtimes in sync cheaply.
				if err := ix.cat.TouchFile(c.rel, prev); err != nil {
					return nil, err
				}
			}
		}
	}

	done := 0
	total := 0
	for _, c := range work {
		if c.changed || c.added {
			total++
		}
	}

	for _, c := range work {
		if !c.changed && !c.added {
			continue
		}
		done++
		if opt.OnProgress != nil {
			opt.OnProgress(done, total, c.rel)
		}
		content, err := os.ReadFile(c.abs)
		if err != nil {
			stats.Failed++
			continue
		}
		units, err := ix.buildUnits(c.rel, c.lang, content, opt.MaxEmbedChars)
		if err != nil {
			stats.Failed++
			continue
		}
		if err := ix.embedUnits(units); err != nil {
			return nil, fmt.Errorf("embed %s: %w", c.rel, err)
		}
		// Documents first, catalog second: a crash in between re-indexes the
		// file next time (ReplaceUnits deletes by path first, so it is
		// idempotent), while the other order would forget the units.
		if err := ix.eng.ReplaceUnits(c.rel, units); err != nil {
			return nil, fmt.Errorf("write %s to infinity: %w", c.rel, err)
		}
		if err := ix.cat.PutFile(c.rel, store.FileState{
			Hash: c.hash, MtimeNs: c.mtimeNs, Size: c.size, Lang: c.lang, Units: len(units),
		}); err != nil {
			return nil, fmt.Errorf("catalog %s: %w", c.rel, err)
		}
		stats.Units += len(units)
		if c.added {
			stats.Added++
		} else {
			stats.Changed++
		}
	}

	stats.Duration = time.Since(start)
	return stats, nil
}

// buildUnits turns one file into searchable units.
func (ix *Indexer) buildUnits(rel, lang string, content []byte, maxChars int) ([]store.Unit, error) {
	res, err := native.Split(lang, rel, content)
	if err != nil {
		return nil, err
	}

	toUnit := func(u native.Unit, code string) store.Unit {
		text := buildEmbedText(rel, lang, u, code, maxChars)
		return store.Unit{
			Path: rel, Lang: lang, Kind: u.Kind, Name: u.Name, Parent: u.Parent,
			StartLine: u.StartLine, EndLine: u.EndLine,
			Text: text, FTS: buildFTSText(u, code), Code: truncate(code, 6000),
			Dim: ix.emb.Dim(),
		}
	}

	var units []store.Unit
	for _, u := range res.Units {
		lo := u.StartByte
		hi := u.EndByte
		if lo < 0 || hi > len(content) || lo >= hi {
			continue
		}
		units = append(units, toUnit(u, string(content[lo:hi])))
	}

	if len(units) == 0 {
		// Unsupported constructs or a file with no declarations: index raw
		// chunks so the file is still findable by content.
		units = append(units, rawUnits(rel, lang, content, ix.emb.Dim())...)
	}
	return units, nil
}

// rawUnits splits a file into fixed windows, used as a fallback.
func rawUnits(rel, lang string, content []byte, dim int) []store.Unit {
	lines := strings.Split(string(content), "\n")
	const window = 80
	var out []store.Unit
	for start := 0; start < len(lines); start += window {
		end := start + window
		if end > len(lines) {
			end = len(lines)
		}
		code := strings.Join(lines[start:end], "\n")
		if strings.TrimSpace(code) == "" {
			continue
		}
		u := native.Unit{
			Kind: "raw", Name: filepath.Base(rel),
			StartLine: start + 1, EndLine: end,
		}
		out = append(out, store.Unit{
			Path: rel, Lang: lang, Kind: u.Kind, Name: u.Name,
			StartLine: u.StartLine, EndLine: u.EndLine,
			Text: buildEmbedText(rel, lang, u, code, 8000),
			FTS:  buildFTSText(u, code), Code: truncate(code, 6000), Dim: dim,
		})
		if len(out) >= 40 {
			break
		}
	}
	return out
}

// buildEmbedText mirrors the layering that works best for code retrieval:
// metadata and signature first, code last, so truncation only drops code tail.
func buildEmbedText(rel, lang string, u native.Unit, code string, maxChars int) string {
	var b strings.Builder
	b.WriteString("File: ")
	b.WriteString(rel)
	b.WriteString("\nLang: ")
	b.WriteString(lang)
	if u.Kind != "" {
		b.WriteString("\nKind: ")
		b.WriteString(u.Kind)
	}
	if u.Name != "" {
		b.WriteString("\nName: ")
		b.WriteString(u.Name)
	}
	if u.Parent != "" {
		b.WriteString("\nParent: ")
		b.WriteString(u.Parent)
	}
	if u.Signature != "" {
		b.WriteString("\nSignature: ")
		b.WriteString(u.Signature)
	}
	if u.Doc != "" {
		b.WriteString("\nDoc: ")
		b.WriteString(u.Doc)
	}
	if len(u.Calls) > 0 {
		b.WriteString("\nCalls: ")
		b.WriteString(strings.Join(u.Calls, " "))
	}
	b.WriteString("\nCode:\n")
	b.WriteString(code)
	s := b.String()
	if len(s) > maxChars {
		s = s[:maxChars]
	}
	return s
}

// buildFTSText pre-splits identifiers so `parse_arguments` and `parseArguments`
// both match a query for `parse`.
func buildFTSText(u native.Unit, code string) string {
	var b strings.Builder
	b.WriteString(SplitIdents(u.Name))
	b.WriteString(" ")
	b.WriteString(SplitIdents(u.Parent))
	b.WriteString(" ")
	b.WriteString(SplitIdents(u.Signature))
	b.WriteString(" ")
	b.WriteString(SplitIdents(u.Doc))
	b.WriteString(" ")
	b.WriteString(strings.Join(u.Calls, " "))
	b.WriteString(" ")
	b.WriteString(SplitIdents(truncate(code, 4000)))
	return b.String()
}

// SplitIdents expands camelCase / snake_case identifiers for the FTS index.
func SplitIdents(s string) string {
	var out strings.Builder
	out.Grow(len(s) + len(s)/3)
	for _, tok := range strings.FieldsFunc(s, func(r rune) bool {
		return !(unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '$' || r == '.')
	}) {
		if tok == "" {
			continue
		}
		out.WriteString(tok)
		out.WriteByte(' ')
		// camelCase / PascalCase boundaries
		var part strings.Builder
		var prev rune
		flush := func() {
			if part.Len() > 0 {
				out.WriteString(strings.ToLower(part.String()))
				out.WriteByte(' ')
				part.Reset()
			}
		}
		runes := []rune(tok)
		for i, r := range runes {
			if r == '_' || r == '.' || r == '$' {
				flush()
				prev = r
				continue
			}
			if i > 0 && unicode.IsUpper(r) {
				lowerBefore := unicode.IsLower(prev) || unicode.IsDigit(prev)
				// Acronym boundary: HTTPResponse -> HTTP | Response
				acronym := unicode.IsUpper(prev) && i+1 < len(runes) && unicode.IsLower(runes[i+1])
				if lowerBefore || acronym {
					flush()
				}
			}
			part.WriteRune(r)
			prev = r
		}
		flush()
	}
	return strings.TrimSpace(out.String())
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max]
}

// embedUnits fills in Mean and Vec for every unit.
func (ix *Indexer) embedUnits(units []store.Unit) error {
	if len(units) == 0 {
		return nil
	}
	texts := make([]string, len(units))
	for i := range units {
		texts[i] = units[i].Text
	}
	vecs, err := ix.emb.EncodeDocs(texts)
	if err != nil {
		return err
	}
	dim := ix.emb.Dim()
	for i := range units {
		units[i].Dim = dim
		units[i].Vec = vecs[i]
	}
	return nil
}
