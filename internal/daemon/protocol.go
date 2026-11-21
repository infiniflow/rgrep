package daemon

import "encoding/json"

// Request is one JSON line from a client.
type Request struct {
	ID     int64           `json:"id"`
	Method string          `json:"method"`
	Params json.RawMessage `json:"params,omitempty"`
}

// Response is one JSON line back to the client.
type Response struct {
	ID     int64           `json:"id"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  string          `json:"error,omitempty"`
}

// IndexParams is the payload for the "index" method.
type IndexParams struct {
	Path  string `json:"path"`
	Force bool   `json:"force"`
}

// SearchParams is the payload for the "search" method.
type SearchParams struct {
	Query    string  `json:"query"`
	K        int     `json:"k"`
	Alpha    float64 `json:"alpha"`
	Path     string  `json:"path,omitempty"`
	Lang     string  `json:"lang,omitempty"`
	Kind     string  `json:"kind,omitempty"`
	NoTests  bool    `json:"no_tests,omitempty"`
	WithCode bool    `json:"with_code,omitempty"`
}

// GetParams is the payload for the "get" method.
type GetParams struct {
	Path string `json:"path"`
	Line int    `json:"line"`
}

// Status is the daemon status payload.
type Status struct {
	Running  bool   `json:"running"`
	PID      int    `json:"pid"`
	Socket   string `json:"socket"`
	Embedder string `json:"embedder"`
	// Infinity is the endpoint that holds the units, and Table the table
	// inside it.
	Infinity string `json:"infinity"`
	Table    string `json:"table"`
	Files    int    `json:"files"`
	Units    int    `json:"units"`
	UptimeS  int64  `json:"uptime_s"`
}
