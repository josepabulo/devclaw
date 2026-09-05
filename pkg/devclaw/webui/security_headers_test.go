package webui

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestCORSDoesNotReflectArbitraryOrigin guards the hole where any site the user
// visited could call this API with their cookie: the middleware echoed back
// whatever Origin arrived, together with Allow-Credentials: true.
func TestCORSDoesNotReflectArbitraryOrigin(t *testing.T) {
	handler := corsMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}), nil)

	cases := []struct {
		origin  string
		allowed bool
	}{
		{"https://attacker.example", false},
		{"https://devclaw.example.com.evil.net", false},
		{"http://localhost:3000", true}, // Vite dev server
		{"http://127.0.0.1:5173", true}, // alternative dev port
	}

	for _, c := range cases {
		t.Run(c.origin, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/api/sessions", nil)
			req.Header.Set("Origin", c.origin)
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)

			got := rec.Header().Get("Access-Control-Allow-Origin")
			if c.allowed && got != c.origin {
				t.Errorf("Allow-Origin = %q, want %q", got, c.origin)
			}
			if !c.allowed && got != "" {
				t.Errorf("Allow-Origin = %q, want empty for an untrusted origin", got)
			}
			if !c.allowed && rec.Header().Get("Access-Control-Allow-Credentials") != "" {
				t.Error("Allow-Credentials leaked to an untrusted origin")
			}
		})
	}
}

func TestCORSHonoursConfiguredOrigins(t *testing.T) {
	handler := corsMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}), []string{"https://painel.exemplo.com"})

	req := httptest.NewRequest(http.MethodGet, "/api/sessions", nil)
	req.Header.Set("Origin", "https://painel.exemplo.com")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "https://painel.exemplo.com" {
		t.Errorf("Allow-Origin = %q, want the configured origin", got)
	}
}

// TestQueryTokenOnlyForEventStream keeps the token out of access logs and
// Referer headers on ordinary routes. EventSource cannot set headers, so the
// query parameter stays available exactly there.
func TestQueryTokenOnlyForEventStream(t *testing.T) {
	t.Run("plain request ignores the query token", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/sessions?token=secret", nil)
		if got := extractToken(req); got != "" {
			t.Errorf("extractToken = %q, want empty on a non-SSE route", got)
		}
	})

	t.Run("event stream accepts the query token", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/chat/stream?token=secret", nil)
		req.Header.Set("Accept", "text/event-stream")
		if got := extractToken(req); got != "secret" {
			t.Errorf("extractToken = %q, want secret (EventSource cannot send headers)", got)
		}
	})

	t.Run("bearer header still wins", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/sessions", nil)
		req.Header.Set("Authorization", "Bearer from-header")
		if got := extractToken(req); got != "from-header" {
			t.Errorf("extractToken = %q, want from-header", got)
		}
	})

	t.Run("cookie still works", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/sessions", nil)
		req.AddCookie(&http.Cookie{Name: "devclaw_token", Value: "from-cookie"})
		if got := extractToken(req); got != "from-cookie" {
			t.Errorf("extractToken = %q, want from-cookie", got)
		}
	})
}
