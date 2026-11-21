//go:build !cgo || (!darwin && !linux)

package infinity

import "fmt"

// Embedded is the engine running inside this process; see embedded.go for the
// real one. Without cgo (or off darwin/linux) rgrep can only talk to a remote
// server, so constructing one is an error rather than a silent fallback.
type Embedded struct{}

// StartEmbedded reports that this build cannot host the engine.
func StartEmbedded(libDir, dataDir, socket string) (*Embedded, error) {
	return nil, fmt.Errorf("the embedded engine needs cgo on darwin or linux; " +
		"set infinity_mode to \"remote\" and point infinity_addr at a server")
}

func (e *Embedded) Socket() string  { return "" }
func (e *Embedded) DataDir() string { return "" }
func (e *Embedded) Lib() string     { return "" }
func (e *Embedded) Stop() error     { return nil }
