import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import axios from "axios";
import "./chat.css";

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const chatEndRef = useRef(null);

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Add user message
    const userMsg = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setTyping(true);

    try {
      const res = await axios.post("http://127.0.0.1:8000/ask", {
        question: input, // ✅ FIXED: must match backend key
        namespace: "user123", // optional, matches your backend
      });

      setTyping(false);
      const botMsg = {
        sender: "bot",
        text: res.data.answer || "No answer found",
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error("Error asking question:", err);
      setTyping(false);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Error: could not fetch response." },
      ]);
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing]);

  return (
    <motion.div
      className="chat-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {/* Chat Box */}
      <div className="chat-box">
        {messages.map((msg, idx) => (
          <motion.div
            key={idx}
            className={`chat-message ${msg.sender}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
          >
            {msg.text}
          </motion.div>
        ))}

        {typing && <p className="chat-typing">🤖 Bot is typing...</p>}
        <div ref={chatEndRef}></div>
      </div>

      {/* Input Section */}
      <div className="chat-input-container">
        <input
          type="text"
          className="chat-input"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <motion.button
          className="chat-send-btn"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={sendMessage}
        >
          ➤
        </motion.button>
      </div>
    </motion.div>
  );
}
