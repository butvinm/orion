WASM_OUT = ../client/orion.wasm

.PHONY: build clean test

build:
	GOOS=js GOARCH=wasm go build -ldflags="-s -w" -o $(WASM_OUT) .

test:
	go test -v -count=1 ./...

clean:
	rm -f $(WASM_OUT)
