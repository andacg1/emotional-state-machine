# The Blue Dahlia Case

An interactive noir interrogation game built with [LangGraph](https://github.com/langchain-ai/langgraph). You play a 1952 LAPD homicide detective questioning Evie Marlowe, a witness whose emotional state shifts dynamically as the conversation unfolds.

The backend is a LangGraph agent graph (`src/agent/graph.ts`) that routes each turn through one of twelve emotional state nodes. The frontend is a React app that streams responses from the LangGraph server in real time.

![Demo 1](static/Blue%20Dahlia%201.png)
![Demo 2](static/Blue%20Dahlia%202.png)

## Prerequisites

- **Node.js** ≥ 20
- **npm**
- A local LLM server compatible with the OpenAI API (e.g. [LM Studio](https://lmstudio.ai/)) **or** an OpenAI API key

## Backend setup

1. Install dependencies:

   ```bash
   npm install
   ```

2. Copy the example environment file and fill in your values:

   ```bash
   cp .env.example .env
   ```

   The key variables are:

   | Variable | Description |
   |---|---|
   | `LMSTUDIO_BASE_URL` | Base URL of your local LLM server (default: `http://localhost:1234/v1`) |
   | `LMSTUDIO_API_KEY` | API key for the local server (default: `lm-studio`) |
   | `LMSTUDIO_MODEL` | Model name to use (e.g. `gemma-4-e4b`) |
   | `LANGSMITH_PROJECT` | (Optional) LangSmith project name for tracing |

3. Start the LangGraph dev server:

   ```bash
   npm run dev
   ```

   The server runs on **http://localhost:2024** by default.

## Frontend setup

1. In a separate terminal, navigate to the `frontend` directory:

   ```bash
   cd frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Start the Vite dev server:

   ```bash
   npm run dev
   ```

   The app opens on **http://localhost:5173**. It proxies all `/api` requests to the LangGraph server at `http://localhost:2024`.

## Running both servers

You need two terminals running simultaneously:

| Terminal | Directory | Command |
|---|---|---|
| Backend | project root | `npm run dev` |
| Frontend | `frontend/` | `npm run dev` |

Open **http://localhost:5173** in your browser to start the interrogation.
