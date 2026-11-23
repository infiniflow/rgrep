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
	"unicode/utf8"

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
	// OnError is called for a file that could not be indexed; may be nil.
	OnError func(path string, err error)
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
		opt.MaxEmbedChars = 1600
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

	// Root guard. The catalog keys files by their **path relative to the root**,
	// so a pass run in a different directory matches almost nothing and the
	// deletion sweep below treats the entire previous index as removed - one
	// `rgrep index .` in the wrong place wipes the index, silently (measured: a
	// 5227-file tree replaced by 25 files from another repo). The files are gone
	// from the catalog by then and only a re-index brings them back, so this
	// refuses instead: name the root, or say `--force` to mean "replace it".
	if len(known) > 0 && !opt.Force {
		overlap := 0
		for rel := range seen {
			if _, ok := known[rel]; ok {
				overlap++
			}
		}
		if overlap*10 < len(known) {
			return nil, fmt.Errorf(
				"refusing to index %s: the index holds %d files from another root "+
					"(%d of them are here); run `rgrep index <that root>` to update it, "+
					"or `rgrep index . --force` to replace the index with this tree",
				root, len(known), overlap)
		}
	}

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

	// Files are encoded a batch at a time. One file is a dozen units, which
	// leaves the model almost nothing to run in parallel: measured 457 units/s
	// encoding file by file against 970 units/s when twenty files are handed
	// over together. The write side stays per file, so one payload the engine
	// refuses still only loses its own file.
	type pendingFile struct {
		c     candFile
		units []store.Unit
	}
	const encodeUnitsPerBatch = 256
	var batch []pendingFile
	batchUnits := 0
	flush := func() error {
		if len(batch) == 0 {
			return nil
		}
		all := make([]store.Unit, 0, batchUnits)
		for _, p := range batch {
			all = append(all, p.units...)
		}
		if err := ix.embedUnits(all); err != nil {
			return fmt.Errorf("embed %d files: %w", len(batch), err)
		}
		off := 0
		for _, p := range batch {
			n := len(p.units)
			units := all[off : off+n]
			off += n
			// Documents first, catalog second: a crash in between re-indexes
			// the file next time (ReplaceUnits deletes by path first, so it is
			// idempotent), while the other order would forget the units.
			//
			// One file the engine refuses does not end the pass: the row is
			// counted in `failed` (which the caller prints) and the file stays
			// out of the catalog, so the next run retries it. Aborting instead
			// made a single payload the engine dislikes - a locale file here -
			// discard a run that had already written 58 k units over half an
			// hour.
			if err := ix.eng.ReplaceUnits(p.c.rel, units); err != nil {
				if opt.OnError != nil {
					opt.OnError(p.c.rel, err)
				}
				stats.Failed++
				continue
			}
			if err := ix.cat.PutFile(p.c.rel, store.FileState{
				Hash: p.c.hash, MtimeNs: p.c.mtimeNs, Size: p.c.size, Lang: p.c.lang, Units: n,
			}); err != nil {
				return fmt.Errorf("catalog %s: %w", p.c.rel, err)
			}
			stats.Units += n
			if p.c.added {
				stats.Added++
			} else {
				stats.Changed++
			}
		}
		batch = batch[:0]
		batchUnits = 0
		return nil
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
		batch = append(batch, pendingFile{c: c, units: units})
		batchUnits += len(units)
		if batchUnits >= encodeUnitsPerBatch {
			if err := flush(); err != nil {
				return nil, err
			}
		}
	}
	if err := flush(); err != nil {
		return nil, err
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
	return truncate(b.String(), maxChars)
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

// truncate cuts s to at most max bytes without splitting a rune.
//
// The cut has to respect runes because the text is a VARCHAR the engine
// validates as UTF-8: a byte cut through a multi-byte character made every
// non-ASCII unit insert fail with `invalid argument: thrift: bad data`
// (measured: 8 of the 8 `web/src/locales/*.ts` files in ragflow, all of them
// CJK/Arabic/Cyrillic, while the ASCII files went through).
func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	cut := max
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut]
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
