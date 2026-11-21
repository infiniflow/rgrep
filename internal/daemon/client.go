package daemon

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"

	"github.com/infiniflow/rgrep/internal/config"
)

// Dial connects to the daemon socket.
func Dial(cfg *config.Config) (net.Conn, error) {
	return net.DialTimeout("unix", cfg.SocketPath, 2*time.Second)
}

// IsRunning reports whether a daemon answers on the socket.
func IsRunning(cfg *config.Config) bool {
	c, err := Dial(cfg)
	if err != nil {
		return false
	}
	c.Close()
	return true
}

// Ensure makes sure a daemon is up, spawning one if necessary.
func Ensure(cfg *config.Config) error {
	if IsRunning(cfg) {
		return nil
	}
	if err := cfg.EnsureDirs(); err != nil {
		return err
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	logPath := LogPath(cfg)
	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		return err
	}
	logf, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer logf.Close()

	cmd := exec.Command(exe, "serve", "--home", cfg.DataDir)
	cmd.Stdout = logf
	cmd.Stderr = logf
	cmd.Stdin = nil
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("spawn daemon: %w", err)
	}
	_ = cmd.Process.Release()

	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if IsRunning(cfg) {
			return nil
		}
		time.Sleep(80 * time.Millisecond)
	}
	return fmt.Errorf("daemon did not start; see %s", logPath)
}

// Call sends one request and decodes the result into out.
func Call(cfg *config.Config, method string, params any, out any) error {
	for attempt := 0; ; attempt++ {
		conn, err := Dial(cfg)
		if err != nil {
			if attempt == 0 {
				if e := Ensure(cfg); e != nil {
					return e
				}
				continue
			}
			return err
		}
		err = callOnce(conn, method, params, out)
		conn.Close()
		if err != nil && attempt == 0 && isRetryable(err) {
			continue
		}
		return err
	}
}

func callOnce(conn net.Conn, method string, params any, out any) error {
	var raw json.RawMessage
	if params != nil {
		b, err := json.Marshal(params)
		if err != nil {
			return err
		}
		raw = b
	}
	req := Request{ID: 1, Method: method, Params: raw}

	enc := json.NewEncoder(conn)
	if err := enc.Encode(&req); err != nil {
		return err
	}
	var resp Response
	dec := json.NewDecoder(bufio.NewReader(conn))
	if err := dec.Decode(&resp); err != nil {
		return err
	}
	if resp.Error != "" {
		return fmt.Errorf("%s", resp.Error)
	}
	if out != nil && len(resp.Result) > 0 {
		return json.Unmarshal(resp.Result, out)
	}
	return nil
}

func isRetryable(err error) bool {
	if err == nil {
		return false
	}
	if _, ok := err.(*net.OpError); ok {
		return true
	}
	return err.Error() == "EOF" || err.Error() == "unexpected EOF"
}
