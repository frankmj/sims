# Makefile for bg_action simulation

all: build

build:
	go build -v

run:
	go run bg_action.go

batch:
	go run bg_action.go -runs 50 -epochs 30

batch-pd:
	go run bg_action.go -runs 50 -epochs 30 -pd

clean:
	rm -f bg_action
