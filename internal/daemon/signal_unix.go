//go:build !windows

package daemon

import "syscall"

var syscallSignalZero = syscall.Signal(0)
