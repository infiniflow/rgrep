NATIVE := native/rgrep_native
BIN := bin/rgrep
# The embedded engine's checkout: `rgrep` loads its shared library at runtime.
ENGINE ?= ../infinity/rust_engine
# Optional: point cargo at a proxy, e.g. CARGO_PROXY=socks5h://127.0.0.1:7891
CARGO_PROXY ?=

CARGO_CONFIG := $(if $(CARGO_PROXY),--config 'http.proxy="$(CARGO_PROXY)"',)

.PHONY: all native engine build test native-test clean ort

all: build

native:
	cd $(NATIVE) && cargo $(CARGO_CONFIG) build --release

# The engine rgrep embeds: one shared library, loaded by the daemon at startup.
engine:
	cd $(ENGINE) && cargo $(CARGO_CONFIG) build --release -p embedded-ffi
	mkdir -p bin/lib
	cp $(ENGINE)/target/release/libinfinity_embedded.dylib bin/lib/ 2>/dev/null || true
	cp $(ENGINE)/target/release/libinfinity_embedded.so bin/lib/ 2>/dev/null || true

$(BIN): native
	mkdir -p bin
	go build -o $(BIN) ./cmd/rgrep
	mkdir -p bin/onnxruntime
	ln -sfn ../../third_party/onnxruntime/lib bin/onnxruntime/lib

build: $(BIN)

native-test:
	cd $(NATIVE) && cargo $(CARGO_CONFIG) test

test: native-test
	go test ./...

# Downloads libonnxruntime from PyPI. Set RGREP_PROXY for a socks5/http proxy.
ort:
	./scripts/fetch_onnxruntime.sh $(ORT_VERSION)

clean:
	rm -rf bin
	cd $(NATIVE) && cargo clean
