function SourceBadge({ sources = [], toolCalls = [] }) {
  const uniqueSources = Array.from(new Set(sources.filter(Boolean)));
  const uniqueTools = Array.from(new Set(toolCalls.filter(Boolean)));

  if (uniqueSources.length === 0 && uniqueTools.length === 0) {
    return null;
  }

  return (
    <div style={styles.container}>
      {uniqueTools.length > 0 && (
        <div style={styles.row}>
          <span style={styles.label}>Tools:</span>
          {uniqueTools.map((tool) => (
            <span key={tool} style={styles.badge}>
              {tool}
            </span>
          ))}
        </div>
      )}

      {uniqueSources.length > 0 && (
        <div style={styles.row}>
          <span style={styles.label}>Sources:</span>
          {uniqueSources.map((source) => (
            <span key={source} style={styles.sourceBadge}>
              {source}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    marginTop: "10px",
    display: "flex",
    flexDirection: "column",
    gap: "6px",
  },
  row: {
    display: "flex",
    flexWrap: "wrap",
    alignItems: "center",
    gap: "6px",
  },
  label: {
    fontSize: "12px",
    fontWeight: 700,
    color: "#4b5563",
  },
  badge: {
    fontSize: "12px",
    padding: "4px 8px",
    borderRadius: "999px",
    background: "#e0f2fe",
    color: "#075985",
    border: "1px solid #bae6fd",
  },
  sourceBadge: {
    fontSize: "12px",
    padding: "4px 8px",
    borderRadius: "999px",
    background: "#ecfdf5",
    color: "#065f46",
    border: "1px solid #bbf7d0",
  },
};

export default SourceBadge;