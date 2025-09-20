import { useState } from "react";
import { motion } from "framer-motion";
import Upload from "./components/Upload";
import Chat from "./components/Chat";
import Signup from "./components/Signup";
import "./app.css";
import DeleteUser from "./components/DeleteUser";

export default function App() {
  const [activeSection, setActiveSection] = useState(null);

  return (
    <div className="app-container">
      <motion.h1
        className="app-title"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        📚 AI Knowledge Assistant
      </motion.h1>

      {/* Buttons */}
      <div className="button-container">
        <motion.button
          className={`switch-btn ${activeSection === "delete" ? "active" : ""}`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setActiveSection("delete")}
        >
          🗑 Delete User
        </motion.button>

        <motion.button
          className={`switch-btn ${activeSection === "upload" ? "active" : ""}`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setActiveSection("upload")}
        >
          📤 Upload PDF
        </motion.button>

        <motion.button
          className={`switch-btn ${activeSection === "chat" ? "active" : ""}`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setActiveSection("chat")}
        >
          💬 Chat
        </motion.button>

        {/* ✅ New Signup Button */}
        <motion.button
          className={`switch-btn ${activeSection === "signup" ? "active" : ""}`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setActiveSection("signup")}
        >
          📝 Signup
        </motion.button>
      </div>

      {/* Section */}
      <motion.div
        className="section-container"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        key={activeSection}
      >
        {activeSection === "upload" && <Upload />}
        {activeSection === "chat" && <Chat />}
        {activeSection === "signup" && <Signup />}
        {activeSection === "delete" && <DeleteUser />}

        {!activeSection && (
          <p className="hint-text">Select an option above to start.</p>
        )}
      </motion.div>
    </div>
  );
}
