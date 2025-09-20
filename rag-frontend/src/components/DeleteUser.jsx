import { useState } from "react";

function DeleteUser() {
  const [username, setUsername] = useState(""); // or namespace
  const [message, setMessage] = useState("");

  const handleDelete = async () => {
    if (!username) {
      setMessage("Please enter a username or namespace");
      return;
    }

    try {
      const url = `http://127.0.0.1:8000/delete-namespace?namespace=${username}`;
      const res = await fetch(url, { method: "DELETE" });
      const data = await res.json();
      setMessage(data.message);
    } catch (error) {
      console.error(error);
      setMessage("Error deleting user/namespace");
    }
  };

  return (
    <div style={{ margin: "20px" }}>
      <h2>Delete User Data</h2>
      <input
        type="text"
        placeholder="Enter username or namespace"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        style={{ padding: "5px", width: "250px", marginRight: "10px" }}
      />
      <button onClick={handleDelete} style={{ padding: "5px 10px" }}>
        Delete
      </button>
      {message && <p>{message}</p>}
    </div>
  );
}

export default DeleteUser;
