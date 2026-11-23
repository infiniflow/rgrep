//go:build cgo && (darwin || linux) && !rgrep_dlopen

package infinity

/*
#cgo LDFLAGS: -L${SRCDIR}/../../bin/lib -linfinity_embedded
#cgo darwin LDFLAGS: -lc++ -framework Security -framework CoreFoundation -liconv
#cgo linux LDFLAGS: -lstdc++ -lpthread -ldl -lm
#include <stdlib.h>

// The engine is linked into this binary (`bin/lib/libinfinity_embedded.a`), so
// these are ordinary externs, not dlsym lookups. No dylib is loaded: the
// archive is the whole engine, and it is the one this build produced.
extern void* inf_embedded_start(const char* data_dir, const char* socket, char** err);
extern int   inf_embedded_stop(void* handle);
extern void  inf_embedded_free(char* s);
*/
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
	"unsafe"
)

// Embedded is the engine running inside this process.
//
// It is statically linked: one binary to ship, no dlopen, no library search
// path, and a version of the engine that is the one the build was made from.
type Embedded struct {
	handle  unsafe.Pointer
	socket  string
	dataDir string
	stopped bool
}

// StartEmbedded opens `dataDir`, serving the engine's thrift front end on
// `socket`. The socket file exists when this returns.
//
// `libDir` is accepted for signature parity with the dlopen build and ignored:
// the engine is already inside the binary.
func StartEmbedded(libDir, dataDir, socket string) (*Embedded, error) {
	_ = libDir
	if err := os.MkdirAll(filepath.Dir(socket), 0o755); err != nil {
		return nil, fmt.Errorf("socket dir: %w", err)
	}
	cDir := C.CString(dataDir)
	defer C.free(unsafe.Pointer(cDir))
	cSock := C.CString(socket)
	defer C.free(unsafe.Pointer(cSock))
	var msg *C.char
	h := C.inf_embedded_start(cDir, cSock, &msg)
	if h == nil {
		text := "unknown error"
		if msg != nil {
			text = C.GoString(msg)
			C.inf_embedded_free(msg)
		}
		return nil, fmt.Errorf("embedded engine: %s", text)
	}
	return &Embedded{handle: h, socket: socket, dataDir: dataDir}, nil
}

// Socket is the unix socket the engine serves, which is what a client dials.
func (e *Embedded) Socket() string { return e.socket }

// DataDir is the directory the engine keeps its segments in.
func (e *Embedded) DataDir() string { return e.dataDir }

// Lib names how the engine was linked: statically, into this binary.
func (e *Embedded) Lib() string { return "static" }

// Stop shuts the engine down and removes its socket.
func (e *Embedded) Stop() error {
	if e == nil || e.stopped {
		return nil
	}
	e.stopped = true
	if rc := C.inf_embedded_stop(e.handle); rc != 0 {
		return fmt.Errorf("embedded engine stop returned %d", int(rc))
	}
	return nil
}
