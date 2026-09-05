package copilot

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"testing"
)

// TestToolPanicIsIsolated is the regression guard for the daemon-killer: a
// panic in any registered tool used to unwind past the executor and take the
// process down, dropping every channel and every in-flight run.
func TestToolPanicIsIsolated(t *testing.T) {
	e := NewToolExecutor(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})))

	e.Register(ToolDefinition{
		Type: "function",
		Function: FunctionDef{
			Name:        "exploding_tool",
			Description: "panics on purpose",
			Parameters:  []byte(`{"type":"object","properties":{}}`),
		},
	}, func(ctx context.Context, args map[string]any) (any, error) {
		// A real crash is an out-of-range index or a nil deref, not an explicit
		// panic — use a genuine runtime fault so the guard is tested honestly.
		parts := make([]string, 0, 1)
		return parts[len(parts)], nil
	})

	call := ToolCall{ID: "call-1", Function: FunctionCall{Name: "exploding_tool", Arguments: "{}"}}

	// Without the recover this line takes the test binary down with it.
	result := e.executeSingle(context.Background(), call)

	if result.Error == nil {
		t.Fatal("Error = nil, want the panic surfaced as a tool error")
	}
	if !strings.Contains(result.Error.Error(), "panicked") {
		t.Errorf("Error = %q, want it to mention the panic", result.Error)
	}
	if result.ToolCallID != "call-1" {
		t.Errorf("ToolCallID = %q, want call-1 (the LLM needs the id to match)", result.ToolCallID)
	}
	if result.Name != "exploding_tool" {
		t.Errorf("Name = %q, want exploding_tool", result.Name)
	}
}

// TestConcurrentToolPanicIsIsolated covers the parallel path, whose goroutines
// have no recover of their own and would crash the process the same way.
func TestConcurrentToolPanicIsIsolated(t *testing.T) {
	e := NewToolExecutor(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})))

	e.Register(ToolDefinition{
		Type: "function",
		Function: FunctionDef{
			Name:        "exploding_tool",
			Description: "panics on purpose",
			Parameters:  []byte(`{"type":"object","properties":{}}`),
		},
	}, func(ctx context.Context, args map[string]any) (any, error) {
		panic("boom")
	})
	e.Register(ToolDefinition{
		Type: "function",
		Function: FunctionDef{
			Name:        "calm_tool",
			Description: "returns normally",
			Parameters:  []byte(`{"type":"object","properties":{}}`),
		},
	}, func(ctx context.Context, args map[string]any) (any, error) {
		return "fine", nil
	})

	calls := []ToolCall{
		{ID: "a", Function: FunctionCall{Name: "exploding_tool", Arguments: "{}"}},
		{ID: "b", Function: FunctionCall{Name: "calm_tool", Arguments: "{}"}},
	}
	results := make([]ToolResult, 2)
	e.executeConcurrentGroup(context.Background(), calls, []int{0, 1}, results, 2)

	if results[0].Error == nil {
		t.Error("panicking tool: Error = nil, want an error")
	}
	// The sibling must still have a result slot filled in, panic or not.
	if results[1].ToolCallID != "b" {
		t.Errorf("sibling ToolCallID = %q, want b", results[1].ToolCallID)
	}
}
