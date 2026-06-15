// =========================================================
// SECTION 01: IMPORTS
// =========================================================

import { useState } from "react";
import axios from "axios";
import SourceBadge from "./SourceBadge";

// =========================================================
// SECTION 02: API CONFIG
// =========================================================

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const CHAT_ENDPOINT = `${API_BASE_URL}/chat`;
const CHAT_STREAM_ENDPOINT = `${API_BASE_URL}/chat/stream`;

function ChatPanel() {
  // =========================================================
  // SECTION 03: STATE MANAGEMENT
  // Purpose:
  // - Track the draft message, history, and request state.
  // =========================================================

  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([
    {
      role: "assistant",
      content:
        "Hello, I'm your Chat Assistant. I can help with title performance, audience engagement, city growth, genre trends, campaign insights, and report-based recommendations. How can I help you today?",
      sources: [],
      tool_calls: [],
    },
  ]);
  const [loading, setLoading] = useState(false);

  // =========================================================
  // SECTION 04: EXAMPLE QUESTIONS
  // =========================================================

  const exampleQuestions = [
    "Which titles performed best in Q1 2025 by watch hours?",
    "Why is Stellar Run trending this month?",
    "Compare audience engagement: Dark Orbit vs Last Kingdom.",
    "Which city had the strongest viewer growth last 30 days?",
    "What explains weak comedy genre performance?",
    "What strategic recommendations would you make for next quarter?",
  ];

  function updateStreamingAssistant(updateFn) {
    setChatHistory((prev) => {
      const next = [...prev];

      for (let index = next.length - 1; index >= 0; index -= 1) {
        const entry = next[index];
        if (entry.role === "assistant" && entry.streaming) {
          next[index] = updateFn(entry);
          return next;
        }
      }

      return next;
    });
  }

  function parseSseEvent(rawEvent) {
    const lines = rawEvent.split("\n");
    let event = "message";
    const dataLines = [];

    for (const line of lines) {
      if (line.startsWith("event:")) {
        event = line.slice(6).trim();
      }

      if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }

    if (dataLines.length === 0) {
      return null;
    }

    try {
      return {
        event,
        data: JSON.parse(dataLines.join("\n")),
      };
    } catch (error) {
      console.error("Unable to parse SSE event:", error);
      return null;
    }
  }

  function handleStreamEvent(streamEvent) {
    if (!streamEvent) return;

    if (streamEvent.event === "status") {
      updateStreamingAssistant((entry) => ({
        ...entry,
        content: streamEvent.data.message || entry.content,
        statusOnly: true,
      }));
      return;
    }

    if (streamEvent.event === "chunk") {
      const delta = streamEvent.data.delta || "";

      updateStreamingAssistant((entry) => ({
        ...entry,
        content: entry.statusOnly ? delta : `${entry.content}${delta}`,
        statusOnly: false,
      }));
      return;
    }

    if (streamEvent.event === "done") {
      updateStreamingAssistant(() => ({
        role: "assistant",
        content: streamEvent.data.answer || "",
        sources: streamEvent.data.sources || [],
        tool_calls: streamEvent.data.tool_calls || [],
        mode: streamEvent.data.mode || "unknown",
        notice: streamEvent.data.notice || null,
        streaming: false,
        statusOnly: false,
      }));
      return;
    }

    if (streamEvent.event === "error") {
      throw new Error(
        streamEvent.data.detail || "Streaming failed while reading backend events.",
      );
    }
  }

  async function sendMessageLegacy(finalMessage) {
    const response = await axios.post(CHAT_ENDPOINT, {
      message: finalMessage,
      session_id: "frontend-session",
      top_k: 4,
    });

    updateStreamingAssistant(() => ({
      role: "assistant",
      content: response.data.answer,
      sources: response.data.sources || [],
      tool_calls: response.data.tool_calls || [],
      mode: response.data.mode || "unknown",
      notice: response.data.notice || null,
      streaming: false,
      statusOnly: false,
    }));
  }

  async function streamMessage(finalMessage) {
    const response = await fetch(CHAT_STREAM_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        message: finalMessage,
        session_id: "frontend-session",
        top_k: 4,
      }),
    });

    if (!response.ok || !response.body) {
      let detail = "Unable to open chat stream.";

      try {
        const payload = await response.json();
        detail = payload.detail || detail;
      } catch (_error) {
        detail = response.statusText || detail;
      }

      throw new Error(detail);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const rawEvents = buffer.split("\n\n");
      buffer = rawEvents.pop() || "";

      for (const rawEvent of rawEvents) {
        const streamEvent = parseSseEvent(rawEvent);
        handleStreamEvent(streamEvent);
      }
    }

    if (buffer.trim()) {
      const streamEvent = parseSseEvent(buffer.trim());
      handleStreamEvent(streamEvent);
    }
  }

  // =========================================================
  // SECTION 05: SEND MESSAGE FUNCTION
  // Purpose:
  // - Call POST /chat and append the grounded response to chat history.
  // =========================================================

  async function sendMessage(customMessage = null) {
    const finalMessage = (customMessage || message).trim();

    if (!finalMessage) return;

    const userEntry = {
      role: "user",
      content: finalMessage,
      sources: [],
      tool_calls: [],
    };

    setMessage("");
    setLoading(true);

    setChatHistory((prev) => [
      ...prev,
      userEntry,
      {
        role: "assistant",
        content: "Thinking and calling tools...",
        sources: [],
        tool_calls: [],
        streaming: true,
        statusOnly: true,
      },
    ]);

    try {
      await streamMessage(finalMessage);
    } catch (error) {
      try {
        await sendMessageLegacy(finalMessage);
      } catch (fallbackError) {
        const errorMessage =
          fallbackError.response?.data?.detail ||
          fallbackError.message ||
          "Unable to get response from backend. Check if FastAPI is running.";

        updateStreamingAssistant(() => ({
          role: "assistant",
          content: `Error: ${errorMessage}`,
          sources: [],
          tool_calls: [],
          streaming: false,
          statusOnly: false,
        }));
      }
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  // =========================================================
  // SECTION 06: UI RENDER
  // =========================================================

  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <h2 style={styles.heading}>Chat Assistant</h2>
        <p style={styles.description}>
          {/* Answers must be grounded using backend tools and source citations. */}
        </p>
      </div>

      <div style={styles.examples}>
        {exampleQuestions.map((question) => (
          <button
            key={question}
            style={styles.exampleButton}
            onClick={() => sendMessage(question)}
            disabled={loading}
          >
            {question}
          </button>
        ))}
      </div>

      <div style={styles.messages}>
        {chatHistory.map((item, index) => (
          <div
            key={`${item.role}-${index}`}
            style={{
              ...styles.message,
              ...(item.role === "user"
                ? styles.userMessage
                : styles.assistantMessage),
            }}
          >
            <div style={styles.role}>
              {item.role === "user" ? "You" : "Assistant"}
            </div>

            <div style={styles.content}>{item.content}</div>

            {item.notice && (
              <div style={styles.notice}>{item.notice}</div>
            )}

            {item.role === "assistant" && (
              <SourceBadge
                sources={item.sources}
                toolCalls={item.tool_calls}
              />
            )}
          </div>
        ))}

      </div>

      <div style={styles.inputArea}>
        <textarea
          value={message}
          placeholder="Ask a DataCore analytics question..."
          onChange={(event) => setMessage(event.target.value)}
          onKeyDown={handleKeyDown}
          style={styles.textarea}
          rows={3}
        />

        <button
          onClick={() => sendMessage()}
          disabled={loading || !message.trim()}
          style={{
            ...styles.sendButton,
            opacity: loading || !message.trim() ? 0.6 : 1,
          }}
        >
          {loading ? "Sending..." : "Send"}
        </button>
      </div>
    </div>
  );
}

// =========================================================
// SECTION 07: STYLES
// =========================================================

const styles = {
  card: {
    height: "100%",
    background: "#ffffff",
    borderRadius: "16px",
    border: "1px solid #e5e7eb",
    boxShadow: "0 10px 25px rgba(15, 23, 42, 0.08)",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  cardHeader: {
    padding: "20px",
    borderBottom: "1px solid #e5e7eb",
  },
  heading: {
    margin: 0,
    fontSize: "22px",
  },
  description: {
    margin: "6px 0 0",
    color: "#6b7280",
    fontSize: "14px",
  },
  examples: {
    padding: "14px 20px",
    display: "flex",
    flexWrap: "wrap",
    gap: "8px",
    borderBottom: "1px solid #e5e7eb",
    background: "#f9fafb",
  },
  exampleButton: {
    border: "1px solid #d1d5db",
    background: "#ffffff",
    borderRadius: "999px",
    padding: "7px 11px",
    fontSize: "12px",
    cursor: "pointer",
  },
  messages: {
    flex: 1,
    minHeight: 0,
    padding: "20px",
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: "14px",
  },
  message: {
    maxWidth: "88%",
    padding: "12px 14px",
    borderRadius: "14px",
    lineHeight: 1.5,
    whiteSpace: "pre-wrap",
  },
  userMessage: {
    alignSelf: "flex-end",
    background: "#111827",
    color: "#ffffff",
  },
  assistantMessage: {
    alignSelf: "flex-start",
    background: "#f3f4f6",
    color: "#111827",
    border: "1px solid #e5e7eb",
  },
  role: {
    fontSize: "12px",
    fontWeight: 700,
    marginBottom: "6px",
    opacity: 0.8,
  },
  content: {
    fontSize: "14px",
  },
  notice: {
    marginTop: "8px",
    fontSize: "12px",
    color: "#92400e",
    background: "#fffbeb",
    border: "1px solid #fde68a",
    borderRadius: "10px",
    padding: "8px 10px",
  },
  inputArea: {
    padding: "16px 20px",
    display: "flex",
    gap: "12px",
    borderTop: "1px solid #e5e7eb",
    background: "#ffffff",
  },
  textarea: {
    flex: 1,
    resize: "none",
    borderRadius: "12px",
    border: "1px solid #d1d5db",
    padding: "12px",
    fontSize: "14px",
    outline: "none",
  },
  sendButton: {
    width: "110px",
    border: "none",
    borderRadius: "12px",
    background: "#2563eb",
    color: "#ffffff",
    fontWeight: 700,
    cursor: "pointer",
  },
};

export default ChatPanel;
