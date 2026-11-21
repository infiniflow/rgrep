module github.com/infiniflow/rgrep

go 1.26.0

require (
	github.com/apache/thrift v0.22.0
	github.com/infiniflow/infinity-go-sdk v0.0.0
	github.com/infiniflow/onnxruntime_go v0.0.0
	modernc.org/sqlite v1.62.0
)

require (
	github.com/dustin/go-humanize v1.1.0 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/mattn/go-isatty v0.0.24 // indirect
	github.com/ncruces/go-strftime v1.1.0 // indirect
	github.com/remyoudompheng/bigfft v0.0.0-20230129092748-24d4a6f8daec // indirect
	golang.org/x/sys v0.49.0 // indirect
	modernc.org/libc v1.80.1 // indirect
	modernc.org/mathutil v1.7.1 // indirect
	modernc.org/memory v1.12.1 // indirect
)

replace github.com/infiniflow/infinity-go-sdk => ../infinity/go

replace github.com/infiniflow/onnxruntime_go => ../onnxruntime_go
