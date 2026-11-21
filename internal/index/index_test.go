package index

import (
	"strings"
	"testing"

	"github.com/infiniflow/rgrep/internal/native"
)

func TestSplitIdents(t *testing.T) {
	cases := map[string][]string{
		"getUserById":   {"getUserById", "get", "user", "by", "id"},
		"parse_arguments": {"parse_arguments", "parse", "arguments"},
		"HTTPResponse":  {"HTTPResponse", "http", "response"},
	}
	for in, want := range cases {
		got := strings.Fields(SplitIdents(in))
		if len(got) == 0 {
			t.Fatalf("%s: no output", in)
		}
		for _, w := range want {
			found := false
			for _, g := range got {
				if g == w {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("%s: missing %q in %v", in, w, got)
			}
		}
	}
}

func TestLanguageFor(t *testing.T) {
	cases := map[string]string{
		"a/b/c.go": "go", "x.py": "python", "y.rs": "rust",
		"z.tsx": "tsx", "q.mjs": "javascript", "README.md": "",
	}
	for p, want := range cases {
		if got := LanguageFor(p); got != want {
			t.Fatalf("%s: got %q want %q", p, got, want)
		}
	}
}

func TestBuildEmbedTextPutsFileBeforeCode(t *testing.T) {
	txt := buildEmbedText("pkg/a.go", "go", native.Unit{Kind: "function", Name: "x"}, "func x() {}", 8000)
	if !strings.HasPrefix(txt, "File: pkg/a.go") {
		t.Fatalf("text must start with File:, got %q", txt)
	}
}
