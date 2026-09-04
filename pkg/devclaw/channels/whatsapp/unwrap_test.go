package whatsapp

import (
	"testing"

	"go.mau.fi/whatsmeow/proto/waE2E"
	"google.golang.org/protobuf/proto"

	"github.com/jholhewres/devclaw/pkg/devclaw/channels"
)

func voiceNote() *waE2E.Message {
	return &waE2E.Message{
		AudioMessage: &waE2E.AudioMessage{
			PTT:      proto.Bool(true),
			Mimetype: proto.String("audio/ogg; codecs=opus"),
			Seconds:  proto.Uint32(7),
		},
	}
}

func wrapFuture(inner *waE2E.Message) *waE2E.FutureProofMessage {
	return &waE2E.FutureProofMessage{Message: inner}
}

// TestExtractMessageContent_UnwrapsContainers covers the failure behind
// "[unsupported message type]" in production: a voice note sent in a
// disappearing chat, as view-once, or edited never reached the audio branch,
// so it was never transcribed.
func TestExtractMessageContent_UnwrapsContainers(t *testing.T) {
	w := createTestWhatsApp()

	cases := []struct {
		name string
		msg  *waE2E.Message
	}{
		{"plain", voiceNote()},
		{"ephemeral", &waE2E.Message{EphemeralMessage: wrapFuture(voiceNote())}},
		{"view once", &waE2E.Message{ViewOnceMessage: wrapFuture(voiceNote())}},
		{"view once v2", &waE2E.Message{ViewOnceMessageV2: wrapFuture(voiceNote())}},
		{"view once v2 ext", &waE2E.Message{ViewOnceMessageV2Extension: wrapFuture(voiceNote())}},
		{"edited", &waE2E.Message{EditedMessage: wrapFuture(voiceNote())}},
		{"device sent", &waE2E.Message{
			DeviceSentMessage: &waE2E.DeviceSentMessage{Message: voiceNote()},
		}},
		{"nested: view once inside ephemeral", &waE2E.Message{
			EphemeralMessage: wrapFuture(&waE2E.Message{
				ViewOnceMessageV2: wrapFuture(voiceNote()),
			}),
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			msg := &channels.IncomingMessage{ID: "x", From: "someone"}
			w.extractMessageContent(tc.msg, msg)

			if msg.Type != channels.MessageAudio {
				t.Fatalf("type = %q, want %q", msg.Type, channels.MessageAudio)
			}
			if msg.Content != "[voice note]" {
				t.Errorf("content = %q, want %q", msg.Content, "[voice note]")
			}
			if msg.Media == nil {
				t.Fatal("media info missing, transcription would have nothing to download")
			}
			if msg.Media.Type != channels.MessageAudio {
				t.Errorf("media type = %q, want %q", msg.Media.Type, channels.MessageAudio)
			}
		})
	}
}

// TestExtractMessageContent_DropsControlAndUnknown pins the other half: a
// control message or a type this build cannot read must leave Type unset so
// handleMessageEvt drops it, instead of being stamped as text and answered.
func TestExtractMessageContent_DropsControlAndUnknown(t *testing.T) {
	w := createTestWhatsApp()

	cases := []struct {
		name string
		msg  *waE2E.Message
	}{
		{"protocol message", &waE2E.Message{
			ProtocolMessage: &waE2E.ProtocolMessage{
				Type: waE2E.ProtocolMessage_REVOKE.Enum(),
			},
		}},
		{"protocol nested in ephemeral", &waE2E.Message{
			EphemeralMessage: wrapFuture(&waE2E.Message{
				ProtocolMessage: &waE2E.ProtocolMessage{
					Type: waE2E.ProtocolMessage_MESSAGE_EDIT.Enum(),
				},
			}),
		}},
		{"empty container", &waE2E.Message{EphemeralMessage: wrapFuture(nil)}},
		{"unknown type", &waE2E.Message{}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			msg := &channels.IncomingMessage{ID: "x", From: "someone"}
			w.extractMessageContent(tc.msg, msg)

			if msg.Type != "" {
				t.Errorf("type = %q, want empty so the caller drops it", msg.Type)
			}
			if msg.Content == "[unsupported message type]" {
				t.Error("the placeholder literal must never be handed to the model as user text")
			}
		})
	}
}

// TestUnwrapMessage_DepthBound makes sure a pathological chain terminates
// instead of recursing without end.
func TestUnwrapMessage_DepthBound(t *testing.T) {
	deep := voiceNote()
	for i := 0; i < maxUnwrapDepth+3; i++ {
		deep = &waE2E.Message{EphemeralMessage: wrapFuture(deep)}
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = unwrapMessage(deep, 0)
	}()
	<-done // panics or hangs here if the bound is broken
}

// TestExtractMessageContent_PlainTextStillWorks guards against the unwrap
// swallowing ordinary messages.
func TestExtractMessageContent_PlainTextStillWorks(t *testing.T) {
	w := createTestWhatsApp()
	msg := &channels.IncomingMessage{ID: "x", From: "someone"}

	w.extractMessageContent(&waE2E.Message{
		Conversation: proto.String("oi, tudo bem?"),
	}, msg)

	if msg.Type != channels.MessageText {
		t.Errorf("type = %q, want %q", msg.Type, channels.MessageText)
	}
	if msg.Content != "oi, tudo bem?" {
		t.Errorf("content = %q", msg.Content)
	}
}
