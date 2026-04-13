import React, { useState } from "react";

function RecommendationList() {
  const [userId, setUserId] = useState("");
  const [items, setItems] = useState([]);

  const fetchRecommendations = async () => {
    try {
      const res = await fetch(`http://127.0.0.1:8000/recommend/${userId}`);
      const data = await res.json();

      setItems(data.recommendations);
    } catch (err) {
      alert("Error fetching recommendations");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h2>🔍 Recommendation System</h2>

      <input
        type="number"
        placeholder="Enter User ID"
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
      />

      <br /><br />

      <button onClick={fetchRecommendations}>
        Get Recommendations
      </button>

      <h3>Results:</h3>

      {/* ✅ FIXED LIST */}
      <ul style={{ listStyle: "none", padding: 0 }}>
        {items.map((item, index) => (
          <li key={index} style={{ margin: "5px" }}>
            Item ID: {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default RecommendationList;