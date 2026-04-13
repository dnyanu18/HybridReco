import React, { useState } from "react";
import axios from "axios";
import RecommendationList from "./RecommendationList";

function Dashboard() {
  const [userId, setUserId] = useState("");
  const [data, setData] = useState(null);

  const getRecommendations = async () => {
    try {
      const res = await axios.get(
        `http://127.0.0.1:8000/recommend/${userId}`
      );
      setData(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div>
      <input
        type="number"
        placeholder="Enter User ID"
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
      />

      <button onClick={getRecommendations}>
        Get Recommendations
      </button>

      {data && <RecommendationList items={data.recommendations} />}
    </div>
  );
}

export default Dashboard;