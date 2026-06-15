// =========================================================
// SECTION 01: IMPORTS
// =========================================================

import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Tooltip,
} from "chart.js";
import { Bar } from "react-chartjs-2";

// =========================================================
// SECTION 02: CHART REGISTRATION
// =========================================================

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

// =========================================================
// SECTION 03: API CONFIG
// =========================================================

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function InsightsChart() {
  // =========================================================
  // SECTION 04: STATE MANAGEMENT
  // Purpose:
  // - Track selected month, genre filter, loaded rows, and loading state.
  // =========================================================

  const [month, setMonth] = useState("2025-05");
  const [selectedGenre, setSelectedGenre] = useState("All");
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);

  // =========================================================
  // SECTION 05: DATA FETCHING
  // Purpose:
  // - Call GET /data/genre-trends for the selected month.
  // =========================================================

  async function fetchGenreTrends() {
    setLoading(true);

    try {
      const response = await axios.get(`${API_BASE_URL}/data/genre-trends`, {
        params: {
          month,
          limit: 20,
        },
      });

      setRows(response.data || []);
    } catch (error) {
      console.error("Failed to fetch genre trends:", error);
      setRows([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchGenreTrends();
  }, [month]);

  // =========================================================
  // SECTION 06: FILTER LOGIC
  // =========================================================

  const genres = useMemo(() => {
    const uniqueGenres = Array.from(new Set(rows.map((row) => row.genre)));
    return ["All", ...uniqueGenres];
  }, [rows]);

  const filteredRows = useMemo(() => {
    if (selectedGenre === "All") {
      return rows;
    }

    return rows.filter((row) => row.genre === selectedGenre);
  }, [rows, selectedGenre]);

  // =========================================================
  // SECTION 07: CHART DATA AND OPTIONS
  // =========================================================

  const chartData = {
    labels: filteredRows.map((row) => row.genre),
    datasets: [
      {
        label: "Watch Hours",
        data: filteredRows.map((row) => row.watch_hours),
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top",
      },
      tooltip: {
        callbacks: {
          afterLabel: function (context) {
            const row = filteredRows[context.dataIndex];

            if (!row) return "";

            return [
              `Avg Completion: ${row.avg_completion_pct}%`,
              `Avg Rating: ${row.avg_rating}`,
              `Sessions: ${row.total_sessions}`,
            ];
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: "Watch Hours",
        },
      },
      x: {
        title: {
          display: true,
          text: "Genre",
        },
      },
    },
  };

  // =========================================================
  // SECTION 08: UI RENDER
  // =========================================================

  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <h2 style={styles.heading}>Genre Performance</h2>
        <p style={styles.description}>
          Chart powered by the backend genre trend endpoint.
        </p>
      </div>

      <div style={styles.filters}>
        <label style={styles.label}>
          Month
          <select
            value={month}
            onChange={(event) => setMonth(event.target.value)}
            style={styles.select}
          >
            <option value="2025-01">2025-01</option>
            <option value="2025-02">2025-02</option>
            <option value="2025-03">2025-03</option>
            <option value="2025-04">2025-04</option>
            <option value="2025-05">2025-05</option>
            <option value="2025-06">2025-06</option>
          </select>
        </label>

        <label style={styles.label}>
          Genre
          <select
            value={selectedGenre}
            onChange={(event) => setSelectedGenre(event.target.value)}
            style={styles.select}
          >
            {genres.map((genre) => (
              <option key={genre} value={genre}>
                {genre}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div style={styles.chartBox}>
        {loading ? (
          <div style={styles.emptyState}>Loading chart...</div>
        ) : filteredRows.length === 0 ? (
          <div style={styles.emptyState}>
            No genre data found for selected filters.
          </div>
        ) : (
          <Bar data={chartData} options={chartOptions} />
        )}
      </div>

      <div style={styles.tableWrapper}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Genre</th>
              <th style={styles.th}>Watch Hours</th>
              <th style={styles.th}>Completion %</th>
              <th style={styles.th}>Rating</th>
            </tr>
          </thead>

          <tbody>
            {filteredRows.map((row) => (
              <tr key={row.genre}>
                <td style={styles.td}>{row.genre}</td>
                <td style={styles.td}>{row.watch_hours}</td>
                <td style={styles.td}>{row.avg_completion_pct}</td>
                <td style={styles.td}>{row.avg_rating}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={styles.sourceNote}>
        Source: SQL tool data from movies, watch_activity, and reviews.
      </div>
    </div>
  );
}

// =========================================================
// SECTION 09: STYLES
// =========================================================

const styles = {
  card: {
    height: "100%",
    background: "#ffffff",
    borderRadius: "16px",
    border: "1px solid #e5e7eb",
    boxShadow: "0 10px 25px rgba(15, 23, 42, 0.08)",
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
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
  filters: {
    padding: "16px 20px",
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "12px",
    background: "#f9fafb",
    borderBottom: "1px solid #e5e7eb",
  },
  label: {
    display: "flex",
    flexDirection: "column",
    gap: "6px",
    fontSize: "13px",
    fontWeight: 700,
    color: "#374151",
  },
  select: {
    padding: "10px",
    borderRadius: "10px",
    border: "1px solid #d1d5db",
    background: "#ffffff",
    fontSize: "14px",
  },
  chartBox: {
    height: "320px",
    padding: "20px",
    flexShrink: 0,
  },
  emptyState: {
    height: "100%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#6b7280",
    border: "1px dashed #d1d5db",
    borderRadius: "12px",
  },
  tableWrapper: {
    padding: "0 20px 20px",
    overflowX: "auto",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: "13px",
  },
  th: {
    textAlign: "left",
    padding: "10px",
    borderBottom: "1px solid #e5e7eb",
    background: "#f9fafb",
  },
  td: {
    padding: "10px",
    borderBottom: "1px solid #f3f4f6",
  },
  sourceNote: {
    marginTop: "auto",
    padding: "14px 20px",
    borderTop: "1px solid #e5e7eb",
    fontSize: "12px",
    color: "#065f46",
    background: "#ecfdf5",
  },
};

export default InsightsChart;
