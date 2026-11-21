package index

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	sdk "github.com/infiniflow/infinity-go-sdk"

	"github.com/infiniflow/rgrep/internal/embed"
	"github.com/infiniflow/rgrep/internal/infinity"
	"github.com/infiniflow/rgrep/internal/store"
)

// TestDebugPayloadRoundTrip builds the real units of one file, writes them
// through the backend, and reads the string columns back.
func TestDebugPayloadRoundTrip(t *testing.T) {
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
		Database: "rgrep_payload", Table: "units", Dim: emb.Dim(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	ix := New(cat, eng, emb)
	units, err := ix.buildUnits("internal/cli/cli.go", "go", content, 8000)
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.embedUnits(units); err != nil {
		t.Fatal(err)
	}
	for i := range units {
		fmt.Printf("send %-16s text=%d fts=%d code=%d\n",
			units[i].Name, len(units[i].Text), len(units[i].FTS), len(units[i].Code))
	}
	if err := eng.ReplaceUnits("internal/cli/cli.go", units); err != nil {
		t.Fatal(err)
	}

	conn, err := sdk.Connect(sdk.NetworkAddress{IP: "127.0.0.1", Port: 23818})
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Disconnect()
	db, err := conn.GetDatabase("rgrep_payload")
	if err != nil {
		t.Fatal(err)
	}
	tbl, err := db.GetTable("units")
	if err != nil {
		t.Fatal(err)
	}
	res, err := tbl.Output([]string{"name", "text", "fts", "code"}).Limit(100).ToResult()
	if err != nil {
		t.Fatal(err)
	}
	qr := res.(*sdk.QueryResult)
	for i := range qr.Data["name"] {
		text, _ := qr.Data["text"][i].(string)
		fts, _ := qr.Data["fts"][i].(string)
		code, _ := qr.Data["code"][i].(string)
		fmt.Printf("got  %-16v text=%d fts=%d code=%d\n",
			qr.Data["name"][i], len(text), len(fts), len(code))
	}
}
