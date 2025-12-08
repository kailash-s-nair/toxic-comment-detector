import { useState } from "react";

const API_URL = "http://localhost:8000/predict";
const LABELS = [
  "toxic",
  "severe_toxic",
  "obscene",
  "threat",
  "insult",
  "identity_hate",
];

function verdictFromScores(scores, threshold) {
  if (!scores) return "—";
  const active = LABELS.filter((label) => (scores[label] ?? 0) >= threshold);
  return active.length > 0 ? "TOXIC" : "Not toxic";
}

function App() {
  const [comment, setComment] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!comment.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comment }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(
        err.message ||
          "Something went wrong. Check that the Python API is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleExample = (text) => {
    setComment(text);
    setResult(null);
    setError("");
  };

  const currentVerdict =
    result && result.scores
      ? verdictFromScores(result.scores, threshold)
      : "—";

  return (
    <div
      style={{
        maxWidth: 900,
        margin: "2rem auto",
        padding: "1.5rem",
        fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
      }}
    >
      <h1 style={{ marginBottom: "0.25rem" }}>Toxic Comment Detector</h1>
      <p style={{ marginTop: 0, color: "#555" }}>
        Web demo powered by your CSCI 4050U project backend.
      </p>

      <section
        style={{
          marginTop: "1.5rem",
          padding: "1rem",
          borderRadius: "0.5rem",
          border: "1px solid #ddd",
          background: "#fafafa",
        }}
      >
        <form onSubmit={handleSubmit}>
          <label
            htmlFor="comment"
            style={{ display: "block", fontWeight: 600, marginBottom: "0.5rem" }}
          >
            Enter a comment
          </label>
          <textarea
            id="comment"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            rows={5}
            style={{
              width: "100%",
              padding: "0.75rem",
              borderRadius: "0.5rem",
              border: "1px solid #ccc",
              resize: "vertical",
              fontFamily: "inherit",
            }}
            placeholder="Type or paste a comment to classify..."
          />

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "1rem",
              marginTop: "0.75rem",
              flexWrap: "wrap",
            }}
          >
            <button
              type="submit"
              disabled={loading || !comment.trim()}
              style={{
                padding: "0.5rem 1.25rem",
                borderRadius: "999px",
                border: "none",
                background: "#2563eb",
                color: "white",
                fontWeight: 600,
                cursor: loading || !comment.trim() ? "not-allowed" : "pointer",
              }}
            >
              {loading ? "Classifying..." : "Classify"}
            </button>

            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <label htmlFor="threshold" style={{ fontSize: "0.9rem" }}>
                Threshold:
              </label>
              <input
                id="threshold"
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={threshold}
                onChange={(e) =>
                  setThreshold(Math.min(1, Math.max(0, Number(e.target.value) || 0)))
                }
                style={{
                  width: "4rem",
                  padding: "0.25rem 0.5rem",
                  borderRadius: "0.25rem",
                  border: "1px solid #55de00ff",
                }}
              />
            </div>
          </div>
        </form>

        <div style={{ marginTop: "0.75rem", fontSize: "0.85rem", color: "#666" }}>
          <span style={{ fontWeight: 600 }}>Examples:</span>{" "}
          <button
            type="button"
            onClick={() =>
              handleExample("I hate you so much, you are the worst.")
            }
            style={{
              border: "none",
              background: "none",
              color: "#2563eb",
              cursor: "pointer",
              textDecoration: "underline",
              padding: 0,
            }}
          >
            toxic insult
          </button>
          {" · "}
          <button
            type="button"
            onClick={() =>
              handleExample("Thank you for your help, I really appreciate it!")
            }
            style={{
              border: "none",
              background: "none",
              color: "#2563eb",
              cursor: "pointer",
              textDecoration: "underline",
              padding: 0,
            }}
          >
            polite message
          </button>
        </div>
      </section>

      {error && (
        <div
          style={{
            marginTop: "1rem",
            padding: "0.75rem 1rem",
            borderRadius: "0.5rem",
            border: "1px solid #fecaca",
            background: "#fef2f2",
            color: "#b91c1c",
          }}
        >
          {error}
        </div>
      )}

      {result && result.scores && (
        <section
          style={{
            marginTop: "1.5rem",
            padding: "1rem",
            borderRadius: "0.5rem",
            border: "1px solid #ddd",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "0.75rem",
              gap: "1rem",
              flexWrap: "wrap",
            }}
          >
            <h2 style={{ margin: 0 }}>Results</h2>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
                fontSize: "0.9rem",
              }}
            >
              <span style={{ fontWeight: 600 }}>Overall verdict:</span>
              <span
                style={{
                  padding: "0.25rem 0.75rem",
                  borderRadius: "999px",
                  background:
                    currentVerdict === "TOXIC" ? "#fee2e2" : "#dcfce7",
                  color:
                    currentVerdict === "TOXIC" ? "#b91c1c" : "#166534",
                  fontWeight: 600,
                }}
              >
                {currentVerdict}
              </span>
            </div>
          </div>

          <p style={{ fontSize: "0.9rem", color: "#555" }}>
            Model: <code>{result.model || "unknown"}</code>
          </p>

          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              marginTop: "0.5rem",
              fontSize: "0.9rem",
            }}
          >
            <thead>
              <tr>
                <th
                  style={{
                    textAlign: "left",
                    borderBottom: "1px solid #e5e7eb",
                    paddingBottom: "0.25rem",
                  }}
                >
                  Label
                </th>
                <th
                  style={{
                    textAlign: "left",
                    borderBottom: "1px solid #e5e7eb",
                    paddingBottom: "0.25rem",
                  }}
                >
                  Score
                </th>
              </tr>
            </thead>
            <tbody>
              {LABELS.map((label) => {
                const score = result.scores[label] ?? 0;
                const isActive = score >= threshold;
                return (
                  <tr key={label}>
                    <td
                      style={{
                        padding: "0.35rem 0",
                        borderBottom: "1px solid #f3f4f6",
                        textTransform: "capitalize",
                      }}
                    >
                      {label.replace("_", " ")}
                    </td>
                    <td
                      style={{
                        padding: "0.35rem 0",
                        borderBottom: "1px solid #f3f4f6",
                      }}
                    >
                      <span
                        style={{
                          padding: "0.15rem 0.5rem",
                          borderRadius: "999px",
                          background: isActive ? "#fee2e2" : "#e5e7eb",
                          color: isActive ? "#b91c1c" : "#374151",
                          fontVariantNumeric: "tabular-nums",
                        }}
                      >
                        {score.toFixed(2)}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <p style={{ marginTop: "0.75rem", fontSize: "0.8rem", color: "#6b7280" }}>
            Scores are probabilities between 0 and 1. Labels above the threshold are
            highlighted as active.
          </p>
        </section>
      )}
    </div>
  );
}

export default App;