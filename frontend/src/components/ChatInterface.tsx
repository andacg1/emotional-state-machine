import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../hooks/useEvieChat";

interface Props {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  onSend: (text: string) => void;
  onReset: () => void;
}

export function ChatInterface({ messages, isLoading, error, onSend, onReset }: Props) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSend(input);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!input.trim() || isLoading) return;
      onSend(input);
      setInput("");
    }
  };

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <span className="chat-title">INTERROGATION TRANSCRIPT</span>
        <button className="reset-btn" onClick={onReset} title="New case">
          [ NEW CASE ]
        </button>
      </div>

      <div className="messages-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-line">— FILE OPEN —</div>
            <div className="empty-sub">
              Victor Vale. Found dead behind the Blue Dahlia Club. 2:15 a.m.<br />
              The singer knows something. Start talking.
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`message message-${msg.role}`}>
            {msg.role === "ai" && (
              <span className="msg-speaker">EVIE &mdash;</span>
            )}
            {msg.role === "human" && (
              <span className="msg-speaker">DETECTIVE &mdash;</span>
            )}
            <span className="msg-text">{msg.content}</span>
          </div>
        ))}

        {isLoading && (
          <div className="message message-ai">
            <span className="msg-speaker">EVIE &mdash;</span>
            <span className="loading-dots">
              <span>.</span><span>.</span><span>.</span>
            </span>
          </div>
        )}

        {error && (
          <div className="error-msg">
            [ CONNECTION ERROR: {error} ]
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <form className="input-area" onSubmit={handleSubmit}>
        <textarea
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask her something, detective..."
          rows={2}
          disabled={isLoading}
        />
        <button
          type="submit"
          className="send-btn"
          disabled={isLoading || !input.trim()}
        >
          SEND
        </button>
      </form>
    </div>
  );
}
