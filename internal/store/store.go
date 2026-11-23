// Package store holds the two things rgrep owns locally: the code-unit
// document model that is written to Infinity, and the SQLite catalog that
// remembers which files have already been indexed.
//
// The catalog is deliberately *not* a search index. Every vector (PLAID
// multi-vector) and full-text (BM25) query runs inside Infinity — see
// package infinity. SQLite keeps only the per-file fingerprint
// (hash / mtime / size / unit count), which is a local filesystem question
// that a database cannot answer for us.
package store

import (
	"database/sql"
	"fmt"
	"sync/atomic"

	_ "modernc.org/sqlite"
)

// FileState is what we remember about a file to detect changes cheaply.
type FileState struct {
	Hash    string
	MtimeNs int64
	Size    int64
	Lang    string
	Units   int
}

// Unit is one searchable code unit. It is the document rgrep writes to
// Infinity: the metadata columns, the text that was embedded, the
// identifier-split text the full-text index tokenises, and the token vectors
// of the ColBERT encoder.
type Unit struct {
	Path      string
	Lang      string
	Kind      string
	Name      string
	Parent    string
	StartLine int
	EndLine   int
	// Text is the structured text that was embedded.
	Text string
	// FTS is Text with identifiers pre-split, for the full-text index.
	FTS string
	// Code is the raw snippet returned to callers.
	Code string
	// Dim is the encoder width. In multi-vector mode Vec is nvec*Dim
	// L2-normalised token vectors; in dense mode it is one Dim-sized
	// L2-normalised vector. The backend writes whichever the column expects.
	Dim int
	Vec []float32
}

// NVec is the number of token vectors in the unit.
func (u *Unit) NVec() int {
	if u.Dim == 0 {
		return 0
	}
	return len(u.Vec) / u.Dim
}

// Catalog is the local incremental state. It records one row per indexed
// file; it holds no vectors and no text.
type Catalog interface {
	Files() (map[string]FileState, error)
	PutFile(path string, fs FileState) error
	TouchFile(path string, fs FileState) error
	DeleteFile(path string) error
	Stats() (files, units int, err error)
	SetMeta(key, value string) error
	GetMeta(key string) string
	Close() error
}

// Store is the SQLite implementation of Catalog.
type Store struct {
	db  *sql.DB
	gen atomic.Int64
}

// schemaVersion 3 dropped the `units` / `units_fts` tables: units live in
// Infinity now, so an old database has nothing this build can use.
const schemaVersion = 3

const schema = `
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS meta (
	key TEXT PRIMARY KEY,
	value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
	path      TEXT PRIMARY KEY,
	hash      TEXT NOT NULL,
	mtime_ns  INTEGER NOT NULL,
	size      INTEGER NOT NULL,
	lang      TEXT NOT NULL,
	units     INTEGER NOT NULL,
	updated_ns INTEGER NOT NULL
);
`

// Open opens (and migrates) the catalog database.
func Open(path string) (*Store, error) {
	dsn := fmt.Sprintf("file:%s?_pragma=busy_timeout(8000)&_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)", path)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)

	var version int
	_ = db.QueryRow(`PRAGMA user_version`).Scan(&version)
	if version != schemaVersion {
		// Layout changed. The catalog is a cache over the filesystem and over
		// Infinity, so dropping it is always safe: the next `rgrep index` pass
		// re-reads every file and re-upserts it into Infinity.
		for _, stmt := range []string{
			`DROP TABLE IF EXISTS units_fts`,
			`DROP TABLE IF EXISTS units`,
			`DROP TABLE IF EXISTS files`,
			`DROP TABLE IF EXISTS meta`,
		} {
			if _, err := db.Exec(stmt); err != nil {
				db.Close()
				return nil, fmt.Errorf("migrate drop: %w", err)
			}
		}
	}
	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}
	if version != schemaVersion {
		if _, err := db.Exec(fmt.Sprintf(`PRAGMA user_version = %d`, schemaVersion)); err != nil {
			db.Close()
			return nil, fmt.Errorf("migrate version: %w", err)
		}
	}
	s := &Store{db: db}
	s.gen.Store(1)
	return s, nil
}

// Close closes the database.
func (s *Store) Close() error { return s.db.Close() }

// Generation increases on every write.
func (s *Store) Generation() int64 { return s.gen.Load() }

func (s *Store) bump() { s.gen.Add(1) }

// SetMeta writes a metadata key.
func (s *Store) SetMeta(key, value string) error {
	_, err := s.db.Exec(`INSERT INTO meta(key, value) VALUES (?, ?)
		ON CONFLICT(key) DO UPDATE SET value = excluded.value`, key, value)
	return err
}

// GetMeta reads a metadata key ("" when absent).
func (s *Store) GetMeta(key string) string {
	var v string
	if err := s.db.QueryRow(`SELECT value FROM meta WHERE key = ?`, key).Scan(&v); err != nil {
		return ""
	}
	return v
}

// Files returns the remembered state of every indexed file.
func (s *Store) Files() (map[string]FileState, error) {
	rows, err := s.db.Query(`SELECT path, hash, mtime_ns, size, lang, units FROM files`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make(map[string]FileState, 1024)
	for rows.Next() {
		var p string
		var fs FileState
		if err := rows.Scan(&p, &fs.Hash, &fs.MtimeNs, &fs.Size, &fs.Lang, &fs.Units); err != nil {
			return nil, err
		}
		out[p] = fs
	}
	return out, rows.Err()
}

// PutFile records that a file's units have been written to Infinity.
func (s *Store) PutFile(path string, fs FileState) error {
	_, err := s.db.Exec(`INSERT INTO files(path, hash, mtime_ns, size, lang, units, updated_ns)
		VALUES (?,?,?,?,?,?, strftime('%s','now')*1000000000)
		ON CONFLICT(path) DO UPDATE SET hash=excluded.hash, mtime_ns=excluded.mtime_ns,
		size=excluded.size, lang=excluded.lang, units=excluded.units, updated_ns=excluded.updated_ns`,
		path, fs.Hash, fs.MtimeNs, fs.Size, fs.Lang, fs.Units)
	if err != nil {
		return err
	}
	s.bump()
	return nil
}

// TouchFile refreshes stat metadata without touching the units.
func (s *Store) TouchFile(path string, fs FileState) error {
	_, err := s.db.Exec(`UPDATE files SET mtime_ns = ?, size = ?, hash = ?, units = ? WHERE path = ?`,
		fs.MtimeNs, fs.Size, fs.Hash, fs.Units, path)
	if err != nil {
		return err
	}
	s.bump()
	return nil
}

// DeleteFile forgets a file (its units are deleted from Infinity separately).
func (s *Store) DeleteFile(path string) error {
	if _, err := s.db.Exec(`DELETE FROM files WHERE path = ?`, path); err != nil {
		return err
	}
	s.bump()
	return nil
}

// Stats returns the number of indexed files and, from those rows, the number
// of units. Both come from the catalog: the units themselves live in Infinity,
// but their counts are recorded here, so `status` needs no round trip.
func (s *Store) Stats() (files, units int, err error) {
	err = s.db.QueryRow(`SELECT COUNT(*), COALESCE(SUM(units), 0) FROM files`).Scan(&files, &units)
	return
}
