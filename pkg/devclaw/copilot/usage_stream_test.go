package copilot

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// sseServer replays a fixed SSE body as an Anthropic-style stream.
func sseServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
}

func testAnthropicClient(t *testing.T, url string) *LLMClient {
	t.Helper()
	return &LLMClient{
		baseURL:    url,
		provider:   "zai-anthropic",
		apiKey:     "test-key",
		model:      "glm-5.3",
		logger:     slog.Default(),
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}
}

// TestAnthropicStream_InputTokensFromMessageDelta covers the telemetry gap seen
// in production: every GLM call logged prompt_tokens=0 with a real
// completion_tokens, so cost was counted as if the prompt were free. The Z.AI
// proxy reports input usage in message_delta, while the parser only read it
// from message_start.
func TestAnthropicStream_InputTokensFromMessageDelta(t *testing.T) {
	// message_start carries no usage at all — the proxy's actual shape.
	body := `event: message_start
data: {"type":"message_start","message":{"id":"m1","role":"assistant","content":[],"usage":{"output_tokens":0}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"oi"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":12345,"output_tokens":185,"cache_read_input_tokens":42}}

event: message_stop
data: {"type":"message_stop"}

`
	srv := sseServer(t, body)
	defer srv.Close()

	c := testAnthropicClient(t, srv.URL)
	resp, err := c.completeOnceStreamAnthropic(context.Background(), "glm-5.3", []chatMessage{
		{Role: "user", Content: "oi"},
	}, nil, func(string) {})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	if resp.Usage.PromptTokens != 12345 {
		t.Errorf("PromptTokens = %d, want 12345 (this is the bug: it was 0)", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 185 {
		t.Errorf("CompletionTokens = %d, want 185", resp.Usage.CompletionTokens)
	}
	if resp.Usage.CacheReadTokens != 42 {
		t.Errorf("CacheReadTokens = %d, want 42", resp.Usage.CacheReadTokens)
	}
}

// TestAnthropicStream_MessageStartStillWins guards the native Anthropic shape:
// when message_start reports the input usage, message_delta must not clobber it.
func TestAnthropicStream_MessageStartStillWins(t *testing.T) {
	body := `event: message_start
data: {"type":"message_start","message":{"id":"m1","role":"assistant","content":[],"usage":{"input_tokens":999,"output_tokens":0,"cache_read_input_tokens":7}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}

event: message_stop
data: {"type":"message_stop"}

`
	srv := sseServer(t, body)
	defer srv.Close()

	c := testAnthropicClient(t, srv.URL)
	resp, err := c.completeOnceStreamAnthropic(context.Background(), "glm-5.3", []chatMessage{
		{Role: "user", Content: "oi"},
	}, nil, func(string) {})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	if resp.Usage.PromptTokens != 999 {
		t.Errorf("PromptTokens = %d, want 999 from message_start", resp.Usage.PromptTokens)
	}
	if resp.Usage.CacheReadTokens != 7 {
		t.Errorf("CacheReadTokens = %d, want 7 from message_start", resp.Usage.CacheReadTokens)
	}
}
