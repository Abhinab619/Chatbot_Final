import { useState, useEffect } from "react";

export default function App() {
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState([]);
  const [userId, setUserId] = useState(localStorage.getItem("user_id") || "");
  const [suggestions, setSuggestions] = useState([]);
  const [searchFocused, setSearchFocused] = useState(false);

  useEffect(() => {
    const storedUserId = localStorage.getItem("user_id");
    if (storedUserId) {
      setUserId(storedUserId);
    }
  }, []);

  const sendMessage = async () => {
    if (!message.trim()) return;
    setChat([...chat, { text: message, sender: "user" }]);
    setMessage("");

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: message, user_id: userId }),
      });

      const data = await response.json();

      if (!userId) {
        setUserId(data.user_id);
        localStorage.setItem("user_id", data.user_id);
      }

      setChat((prev) => [...prev, { text: data.response, sender: "bot" }]);

      if (data.recommended_question) {
        const parsedSuggestions = data.recommended_question
          .split("\n")
          .map((q) => q.replace(/^\d+\.\s*/, "").trim())
          .filter(Boolean);
        setSuggestions(parsedSuggestions);
      }
    } catch (error) {
      console.error("Error:", error);
      setChat((prev) => [...prev, { text: "Error getting response.", sender: "bot" }]);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setMessage(suggestion);
    setSearchFocused(false);
    setTimeout(() => sendMessage(), 100); // auto-send after suggestion click
  };

  return (
    <div style={styles.pageContainer}>
      <header style={styles.header}>UDYAMI HELPDESK</header>

      <div style={styles.chatContainer}>
        <div style={styles.chatWindow}>
          {chat.map((msg, index) => (
            <div
              key={index}
              style={{
                ...styles.message,
                alignSelf: msg.sender === "user" ? "flex-end" : "flex-start",
                backgroundColor: msg.sender === "user" ? "#007bff" : "#e0e0e0",
                color: msg.sender === "user" ? "#fff" : "#000",
              }}
            >
              {msg.text}
            </div>
          ))}
        </div>

        <div style={styles.inputContainer}>
          <input
            type="text"
            style={styles.input}
            placeholder="Type a message..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            onFocus={() => setSearchFocused(true)}
            onBlur={() => {
              setTimeout(() => {
                const active = document.activeElement;
                if (!active.closest(".suggestion-box")) {
                  setSearchFocused(false);
                }
              }, 150);
            }}
          />
          <button onClick={sendMessage} style={styles.button}>Send</button>
        </div>

        {searchFocused && suggestions.length > 0 && (
          <div className="suggestion-box" style={styles.suggestionBox}>
            <label style={{ fontSize: "12px", color: "#555" }}>Suggestions:</label>
            <ul style={styles.suggestionList}>
              {suggestions.map((s, i) => (
                <li
                  key={i}
                  style={styles.suggestionItem}
                  onClick={() => handleSuggestionClick(s)}
                  tabIndex={0}
                >
                  {s}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  pageContainer: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    height: "100vh",
    backgroundColor: "#f4f4f9",
  },
  header: {
    fontSize: "24px",
    fontWeight: "bold",
    color: "#333",
    padding: "15px",
  },
  chatContainer: {
    width: "400px",
    backgroundColor: "#ffffff",
    boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)",
    borderRadius: "10px",
    overflow: "hidden",
  },
  chatWindow: {
    height: "400px",
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    padding: "15px",
    gap: "10px",
    backgroundColor: "#f9f9f9",
  },
  message: {
    maxWidth: "70%",
    padding: "10px",
    borderRadius: "8px",
  },
  inputContainer: {
    display: "flex",
    alignItems: "center",
    padding: "10px",
    backgroundColor: "#fff",
    borderTop: "1px solid #ddd",
  },
  input: {
    flex: 1,
    padding: "8px",
    border: "1px solid #ccc",
    borderRadius: "5px",
    outline: "none",
  },
  button: {
    marginLeft: "10px",
    backgroundColor: "#007bff",
    color: "white",
    padding: "8px 15px",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  },
  suggestionBox: {
    padding: "10px",
    backgroundColor: "#f1f1f1",
    borderTop: "1px solid #ddd",
    maxHeight: "150px",
    overflowY: "auto",
  },
  suggestionList: {
    listStyle: "none",
    paddingLeft: "10px",
    marginTop: "5px",
    marginBottom: 0,
  },
  suggestionItem: {
    padding: "5px 0",
    cursor: "pointer",
    color: "#007bff",
    fontSize: "14px",
  },
};
