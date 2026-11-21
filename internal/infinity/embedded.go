//go:build cgo && (darwin || linux)

package infinity

/*
#include <dlfcn.h>
#include <stdlib.h>

// The functions below exist so Go never has to cast a dlsym result to a Go
// function pointer.

typedef void* (*rg_start_fn)(const char*, const char*, char**);
typedef int   (*rg_stop_fn)(void*);
typedef void  (*rg_free_fn)(char*);

static void* rg_dlopen(const char* path, char** err) {
	void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
	if (h == NULL) {
		const char* e = dlerror();
		*err = (char*)(e != NULL ? e : "dlopen failed");
	}
	return h;
}

static void* rg_dlsym(void* handle, const char* name) {
	return dlsym(handle, name);
}

static void* rg_call_start(void* fn, const char* data_dir, const char* socket, char** err) {
	return ((rg_start_fn)fn)(data_dir, socket, err);
}

static int rg_call_stop(void* fn, void* handle) {
	return ((rg_stop_fn)fn)(handle);
}

static void rg_call_free(void* fn, char* s) {
	((rg_free_fn)fn)(s);
}
*/
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
	"unsafe"
)

// libNames is what a lib dir may contain, in preference order.
var libNames = []string{"libinfinity_embedded.dylib", "libinfinity_embedded.so"}

// Embedded is the engine running inside this process.
//
// It exists because a local client should not have to install a database, pick
// a port and keep a supervisor alive: the engine is loaded from a shared
// library, serves the ordinary thrift API on a unix socket in the caller's data
// directory, and stops when the process does. Nothing outside the process can
// reach it, and nothing has to outlive it.
type Embedded struct {
	handle  unsafe.Pointer
	stopFn  unsafe.Pointer
	freeFn  unsafe.Pointer
	socket  string
	dataDir string
	libPath string
	stopped bool
}

// StartEmbedded loads the engine and opens `dataDir`, serving on `socket`.
//
// `libDir` is scanned for libinfinity_embedded.{dylib,so}; a path to the library
// itself is accepted too. The socket file exists when this returns.
func StartEmbedded(libDir, dataDir, socket string) (*Embedded, error) {
	libPath, err := findLib(libDir)
	if err != nil {
		return nil, err
	}

	// The library is never `dlclose`d: the engine registers statics (allocator,
	// thread names, panic hooks) whose destructors are not safe to run while the
	// process is still alive, and a host that unloads and reloads it gains
	// nothing. Keeping it mapped costs address space, not threads.
	var cerr *C.char
	cpath := C.CString(libPath)
	defer C.free(unsafe.Pointer(cpath))
	handle := C.rg_dlopen(cpath, &cerr)
	if handle == nil {
		return nil, fmt.Errorf("load %s: %s", libPath, C.GoString(cerr))
	}
	sym := func(name string) (unsafe.Pointer, error) {
		cname := C.CString(name)
		defer C.free(unsafe.Pointer(cname))
		p := C.rg_dlsym(handle, cname)
		if p == nil {
			return nil, fmt.Errorf("%s has no %s (built without the embedded engine?)", libPath, name)
		}
		return p, nil
	}
	startFn, err := sym("inf_embedded_start")
	if err != nil {
		return nil, err
	}
	stopFn, err := sym("inf_embedded_stop")
	if err != nil {
		return nil, err
	}
	freeFn, err := sym("inf_embedded_free")
	if err != nil {
		return nil, err
	}

	if err := os.MkdirAll(filepath.Dir(socket), 0o755); err != nil {
		return nil, fmt.Errorf("socket dir: %w", err)
	}
	cDir := C.CString(dataDir)
	defer C.free(unsafe.Pointer(cDir))
	cSock := C.CString(socket)
	defer C.free(unsafe.Pointer(cSock))
	var msg *C.char
	h := C.rg_call_start(startFn, cDir, cSock, &msg)
	if h == nil {
		text := "unknown error"
		if msg != nil {
			text = C.GoString(msg)
			C.rg_call_free(freeFn, msg)
		}
		return nil, fmt.Errorf("embedded engine: %s", text)
	}
	return &Embedded{
		handle:  h,
		stopFn:  stopFn,
		freeFn:  freeFn,
		socket:  socket,
		dataDir: dataDir,
		libPath: libPath,
	}, nil
}

// Socket is the unix socket the engine serves, which is what a client dials.
func (e *Embedded) Socket() string { return e.socket }

// DataDir is the directory the engine keeps its segments in.
func (e *Embedded) DataDir() string { return e.dataDir }

// Lib is the shared library that was loaded.
func (e *Embedded) Lib() string { return e.libPath }

// Stop shuts the engine down and removes its socket.
//
// The data directory is closed and reopenable afterwards, so a caller may stop
// one engine and start another in the same process (tests do). Calling it twice
// is a no-op.
func (e *Embedded) Stop() error {
	if e == nil || e.stopped {
		return nil
	}
	e.stopped = true
	if rc := C.rg_call_stop(e.stopFn, e.handle); rc != 0 {
		return fmt.Errorf("embedded engine stop returned %d", int(rc))
	}
	return nil
}

// findLib resolves the shared library: either the dir holding it, or the file.
func findLib(libDir string) (string, error) {
	if libDir == "" {
		return "", fmt.Errorf("no lib dir for the embedded engine: set infinity_lib_dir")
	}
	if st, err := os.Stat(libDir); err == nil && !st.IsDir() {
		return libDir, nil
	}
	for _, name := range libNames {
		p := filepath.Join(libDir, name)
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p, nil
		}
	}
	return "", fmt.Errorf("no %v in %s (build it with `make engine`)", libNames, libDir)
}
