import "./App.css";
import { AsciiSprite } from "./components/AsciiSprite";
import { ChatInterface } from "./components/ChatInterface";
import { useEvieChat } from "./hooks/useEvieChat";

export default function App() {
  const { messages, evieState, isLoading, error, sendMessage, resetThread } = useEvieChat();

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <span className="header-rule">&#9472;&#9472;&#9472;</span>
        </div>
        <div className="header-title">
          <span className="header-precinct">LAPD · HOMICIDE · 1952</span>
          <h1 className="header-case">THE BLUE DAHLIA CASE</h1>
        </div>
        <div className="header-right">
          <span className="header-rule">&#9472;&#9472;&#9472;</span>
        </div>
      </header>

      <main className="app-main">
        <AsciiSprite
          node={evieState.currentNode}
          milestones={evieState.milestones}
          trustLevel={evieState.trustLevel}
          fearLevel={evieState.fearLevel}
        />
        <ChatInterface
          messages={messages}
          isLoading={isLoading}
          error={error}
          onSend={sendMessage}
          onReset={resetThread}
        />
      </main>

      <footer className="app-footer">
        <span>CONFIDENTIAL — FOR OFFICIAL USE ONLY — LAPD RECORDS DIV.</span>
      </footer>
    </div>
  );
}
