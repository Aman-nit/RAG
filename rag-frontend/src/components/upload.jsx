import { motion } from "framer-motion";
import { useState } from "react";
import axios from "axios";
import "./upload.css";
import "./navbar.jsx";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  const handleUpload = async () => {
    if (!file) return alert("Please select a file first!");
    const formData = new FormData();
    formData.append("file", file);

    try {
      setIsUploading(true);
      setStatus("Uploading...");
      await axios.post(
        "http://127.0.0.1:8000/upload?namespace=user123",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setStatus("✅ File uploaded successfully!");
    } catch (err) {
      setStatus("❌ Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <motion.div
      className="upload-container"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* File Upload Card */}
      <motion.div
        className="upload-box"
        whileHover={{ scale: 1.03 }}
        onClick={() => document.getElementById("fileInput").click()}
      >
        <input
          type="file"
          id="fileInput"
          className="hidden"
          onChange={(e) => setFile(e.target.files[0])}
        />
        {file ? (
          <p className="file-name">{file.name}</p>
        ) : (
          <p className="file-placeholder">📂 Drag & Drop or Click to Select</p>
        )}
      </motion.div>

      {/* Upload Button */}
      <motion.button
        className={`upload-btn ${isUploading ? "disabled" : ""}`}
        whileHover={!isUploading ? { scale: 1.05 } : {}}
        whileTap={!isUploading ? { scale: 0.95 } : {}}
        onClick={handleUpload}
        disabled={isUploading}
      >
        {isUploading ? "Uploading..." : "Upload"}
      </motion.button>

      {/* Status Message */}
      {status && (
        <motion.p
          className={`status-msg ${
            status.includes("✅")
              ? "success"
              : status.includes("❌")
              ? "error"
              : "info"
          }`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          {status}
        </motion.p>
      )}
    </motion.div>
  );
}
