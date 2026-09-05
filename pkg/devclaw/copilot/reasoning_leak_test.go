package copilot

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// anthropicThinkingStream fakes the SSE a reasoning model returns over the
// Anthropic Messages API — the shape GLM-5.3 produces, since it cannot turn
// reasoning off.
func anthropicThinkingStream(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		frames := []string{
			`event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`,

			`event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}`,

			`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"The user asks for an overview. Let me check memory first."}}`,

			`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" The user communicates in Portuguese, so respond in PT-BR."}}`,

			`event: content_block_stop
data: {"type":"content_block_stop","index":0}`,

			`event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text"}}`,

			`event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Overview do momento: tudo certo."}}`,

			`event: content_block_stop
data: {"type":"content_block_stop","index":1}`,

			`event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}`,
		}
		for _, f := range frames {
			fmt.Fprintf(w, "%s\n\n", f)
			if flusher != nil {
				flusher.Flush()
			}
		}
	}))
}

// TestAnthropicReasoningNeverReachesUser is the regression guard for reasoning
// being delivered to WhatsApp as one message per thought. The stream may carry
// reasoning, but only wrapped, so the tag-based filters downstream can remove
// it before a channel sends anything.
func TestAnthropicReasoningNeverReachesUser(t *testing.T) {
	srv := anthropicThinkingStream(t)
	defer srv.Close()

	c := &LLMClient{
		provider:   "anthropic",
		baseURL:    srv.URL,
		model:      "glm-5.3",
		httpClient: srv.Client(),
		logger:     slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
	}

	var streamed strings.Builder
	resp, err := c.completeOnceStreamAnthropic(
		context.Background(), "glm-5.3",
		[]chatMessage{{Role: "user", Content: "me dá um overview"}},
		nil,
		func(chunk string) { streamed.WriteString(chunk) },
	)
	if err != nil {
		t.Fatalf("stream failed: %v", err)
	}

	got := streamed.String()

	t.Run("reasoning is wrapped, not bare", func(t *testing.T) {
		if !strings.Contains(got, "<thinking>") || !strings.Contains(got, "</thinking>") {
			t.Errorf("stream is missing the thinking wrapper, so nothing downstream can filter it:\n%s", got)
		}
	})

	t.Run("stripping the tags removes every thought", func(t *testing.T) {
		clean := StripInternalTags(got)
		for _, leaked := range []string{
			"Let me check memory",
			"The user asks",
			"respond in PT-BR",
		} {
			if strings.Contains(clean, leaked) {
				t.Errorf("reasoning survived sanitisation: %q still present in %q", leaked, clean)
			}
		}
	})

	t.Run("the real answer survives", func(t *testing.T) {
		clean := StripInternalTags(got)
		if !strings.Contains(clean, "Overview do momento") {
			t.Errorf("the user-visible answer was lost: %q", clean)
		}
	})

	t.Run("reasoning stays out of the assistant message", func(t *testing.T) {
		if strings.Contains(resp.Content, "Let me check memory") {
			t.Errorf("reasoning leaked into the final content: %q", resp.Content)
		}
		if !strings.Contains(resp.Content, "Overview do momento") {
			t.Errorf("final content lost the answer: %q", resp.Content)
		}
	})
}

// TestThinkingWrapperClosesOnTruncatedStream covers a stream that dies mid
// thought: an unterminated wrapper would slip past the tagged-block filter.
func TestThinkingWrapperClosesOnTruncatedStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, `event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"halfway through a thought"}}

`)
		if flusher != nil {
			flusher.Flush()
		}
		// Stream ends here — no content_block_stop, no text, no message_delta.
	}))
	defer srv.Close()

	c := &LLMClient{
		provider:   "anthropic",
		baseURL:    srv.URL,
		model:      "glm-5.3",
		httpClient: srv.Client(),
		logger:     slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
	}

	var streamed strings.Builder
	if _, err := c.completeOnceStreamAnthropic(
		context.Background(), "glm-5.3",
		[]chatMessage{{Role: "user", Content: "oi"}}, nil,
		func(chunk string) { streamed.WriteString(chunk) },
	); err != nil {
		t.Fatalf("stream failed: %v", err)
	}

	got := streamed.String()
	if strings.Count(got, "<thinking>") != strings.Count(got, "</thinking>") {
		t.Errorf("unbalanced thinking wrapper on a truncated stream: %q", got)
	}
	if clean := StripInternalTags(got); strings.Contains(clean, "halfway through a thought") {
		t.Errorf("truncated reasoning survived sanitisation: %q", clean)
	}
}
