package cli

import (
	"fmt"
	"os"

	"github.com/infiniflow/rgrep/internal/config"
	"github.com/infiniflow/rgrep/internal/daemon"
	"github.com/infiniflow/rgrep/internal/native"
)

var abiValue = native.ABI()

func serve(cfg *config.Config) int {
	if err := daemon.Serve(cfg, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "serve:", err)
		return 1
	}
	return 0
}
