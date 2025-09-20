import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <nav className="bg-gray-900 text-white px-6 py-4 flex justify-between items-center shadow-md">
      <h1 className="text-2xl font-bold">RAG App</h1>
      <div className="flex gap-6">
        <Link to="/" className="hover:text-blue-400 transition">
          Upload
        </Link>
        <Link to="/chat" className="hover:text-blue-400 transition">
          Chat
        </Link>
      </div>
    </nav>
  );
}
