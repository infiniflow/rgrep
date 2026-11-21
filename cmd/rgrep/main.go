package main

import (
	"os"

	"github.com/infiniflow/rgrep/internal/cli"
)

func main() {
	os.Exit(cli.Run(os.Args[1:]))
}
