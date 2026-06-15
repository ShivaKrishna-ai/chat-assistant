// =========================================================
// SECTION 01: IMPORTS
// =========================================================

import "./App.css";
import ChatPanel from "./components/ChatPanel";
import InsightsChart from "./components/InsightsChart";

// =========================================================
// SECTION 02: MAIN APP LAYOUT
// Purpose:
// - Render the left chat panel and right analytics panel.
// =========================================================

function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <h1 className="app-title">Chat Assistant</h1>
        <p className="app-subtitle">
          SQL + RAG powered internal analytics assistant with source attribution
        </p>
      </header>

      <main className="app-main">
        <section className="app-panel app-panel--chat">
          <ChatPanel />
        </section>

        <section className="app-panel app-panel--insights">
          <InsightsChart />
        </section>
      </main>
    </div>
  );
}

export default App;
