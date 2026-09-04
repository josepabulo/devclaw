package telegram

import (
	"errors"
	"fmt"
	"net/url"
	"strings"
	"testing"
)

// TestRedactTokenHidesCredentialFromTransportErrors covers the leak seen in
// production: a getUpdates 502 logged the full request URL, and the bot token
// is part of that path, so the credential landed in a log file with no
// rotation.
func TestRedactTokenHidesCredentialFromTransportErrors(t *testing.T) {
	const token = "8617429154:AAHTdYf9g59MB7Vj-TBxvEu667047kWgXBk"
	tg := &Telegram{cfg: Config{Token: token}}

	// The shape net/http actually returns for a failed request.
	transportErr := &url.Error{
		Op:  "Post",
		URL: "https://api.telegram.org/bot" + token + "/getUpdates",
		Err: errors.New("context deadline exceeded"),
	}

	wrapped := fmt.Errorf("telegram: %s request failed: %w", "getUpdates", tg.redactToken(transportErr))

	if strings.Contains(wrapped.Error(), token) {
		t.Errorf("token leaked into error text: %s", wrapped.Error())
	}
	if !strings.Contains(wrapped.Error(), "<redacted>") {
		t.Errorf("expected redaction marker, got: %s", wrapped.Error())
	}
	// The cause must stay reachable for callers that classify errors.
	if !errors.Is(wrapped, transportErr) {
		t.Error("original error no longer reachable via errors.Is")
	}
	var asURLErr *url.Error
	if !errors.As(wrapped, &asURLErr) {
		t.Error("original error no longer reachable via errors.As")
	}
}

func TestRedactTokenPassesThroughUnrelatedErrors(t *testing.T) {
	tg := &Telegram{cfg: Config{Token: "secret-token"}}

	orig := errors.New("connection reset by peer")
	if got := tg.redactToken(orig); got != orig {
		t.Errorf("error without the token should be returned unchanged, got %v", got)
	}
	if tg.redactToken(nil) != nil {
		t.Error("nil must stay nil")
	}

	// An empty token must not turn every error into a redaction of "".
	empty := &Telegram{cfg: Config{Token: ""}}
	if got := empty.redactToken(orig); got != orig {
		t.Errorf("empty token must be a no-op, got %v", got)
	}
}
