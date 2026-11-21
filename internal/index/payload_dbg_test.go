package index

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	sdk "github.com/infiniflow/infinity-go-sdk"

	"github.com/infiniflow/rgrep/internal/embed"
	"github.com/infiniflow/rgrep/internal/infinity"
	"github.com/infiniflow/rgrep/internal/store"
)

// TestDebugPayloadVariant writes the same units three ways: as built, with
// ASCII-only stand-ins of the same length, and with short values. Whichever
// variant survives tells us whether the content or the shape is at fault.
func TestDebugPayloadVariant(t *testing.T) {
	content, err := os.ReadFile("../../internal/cli/cli.go")
	if err != nil {
		t.Fatal(err)
	}
	cat, err := store.Open(filepath.Join(t.TempDir(), "c.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer cat.Close()

	emb := embed.NewHash(64)
	eng, err := infinity.Open(infinity.Config{
		Host: "127.0.0.1", Port: 23818,
		Database: "rgrep_variant", Table: "units", Dim: emb.Dim(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	ix := New(cat, eng, emb)
	base, err := ix.buildUnits("internal/cli/cli.go", "go", content, 8000)
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.embedUnits(base); err != nil {
		t.Fatal(err)
	}

	var rows []store.Unit
	for _, u := range base {
		rows = append(rows, u)

		ascii := u
		ascii.Path = u.Path + ".ascii"
		ascii.Name = u.Name + "A"
		ascii.Text = "T" + strings.Repeat("t", len(u.Text)-1)
		ascii.FTS = "F" + strings.Repeat("f", len(u.FTS)-1)
		ascii.Code = "C" + strings.Repeat("c", len(u.Code)-1)
		rows = append(rows, ascii)

		short := u
		short.Path = u.Path + ".short"
		short.Name = u.Name + "S"
		short.Text = "TEXT-SHORT"
		short.FTS = "FTS-SHORT"
		short.Code = "CODE-SHORT"
		rows = append(rows, short)
	}
	if err := eng.ReplaceUnits("internal/cli/cli.go", rows); err != nil {
		t.Fatal(err)
	}

	conn, err := sdk.Connect(sdk.NetworkAddress{IP: "127.0.0.1", Port: 23818})
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Disconnect()
	db, err := conn.GetDatabase("rgrep_variant")
	if err != nil {
		t.Fatal(err)
	}
	tbl, err := db.GetTable("units")
	if err != nil {
		t.Fatal(err)
	}
	res, err := tbl.Output([]string{"name", "text", "fts", "code"}).Limit(300).ToResult()
	if err != nil {
		t.Fatal(err)
	}
	qr := res.(*sdk.QueryResult)
	stats := map[string][2]int{} // variant -> {bad, total}
	for i := range qr.Data["name"] {
		name, _ := qr.Data["name"][i].(string)
		text, _ := qr.Data["text"][i].(string)
		fts, _ := qr.Data["fts"][i].(string)
		code, _ := qr.Data["code"][i].(string)
		variant := "real"
		switch {
		case strings.HasSuffix(name, "A"):
			variant = "ascii"
		case strings.HasSuffix(name, "S"):
			variant = "short"
		}
		var ok bool
		switch variant {
		case "ascii":
			ok = strings.HasPrefix(text, "T") && strings.HasPrefix(fts, "F") && strings.HasPrefix(code, "C")
		case "short":
			ok = text == "TEXT-SHORT" && fts == "FTS-SHORT" && code == "CODE-SHORT"
		default:
			want, found := lookup(base, strings.TrimSuffix(name, "A"))
			if !found {
				continue
			}
			ok = text == want.Text && fts == want.FTS && code == want.Code
			if !ok && stats["real"][1] < 3 {
				fmt.Printf("real %-12s text %d/%d fts %d/%d code %d/%d\n",
					name, len(text), len(want.Text), len(fts), len(want.FTS), len(code), len(want.Code))
			}
		}
		s := stats[variant]
		s[1]++
		if !ok {
			s[0]++
		}
		stats[variant] = s
	}
	for _, v := range []string{"real", "ascii", "short"} {
		fmt.Printf("%-6s bad=%d total=%d\n", v, stats[v][0], stats[v][1])
	}
}

func lookup(units []store.Unit, name string) (store.Unit, bool) {
	for _, u := range units {
		if u.Name == name {
			return u, true
		}
	}
	return store.Unit{}, false
}
