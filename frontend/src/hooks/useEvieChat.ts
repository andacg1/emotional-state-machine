import { useState, useRef, useCallback } from "react";
import { Client } from "@langchain/langgraph-sdk";
import type { EvieNode } from "../sprites";

export interface ChatMessage {
  id: string;
  role: "human" | "ai";
  content: string;
}

export interface EvieState {
  currentNode: EvieNode;
  emotion: string;
  milestones: string[];
  trustLevel: number;
  fearLevel: number;
}

const DEFAULT_STATE: EvieState = {
  currentNode: "POLITE_MASK",
  emotion: "relaxed",
  milestones: [],
  trustLevel: 1,
  fearLevel: 0,
};

const client = new Client({ apiUrl: `${window.location.origin}/api` });

export function useEvieChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [evieState, setEvieState] = useState<EvieState>(DEFAULT_STATE);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const threadIdRef = useRef<string | null>(null);

  const sendMessage = useCallback(async (userInput: string) => {
    if (!userInput.trim() || isLoading) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "human",
      content: userInput.trim(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);
    setError(null);

    try {
      if (!threadIdRef.current) {
        const thread = await client.threads.create();
        threadIdRef.current = thread.thread_id;
      }

      const stream = client.runs.stream(threadIdRef.current, "agent", {
        input: { messages: [{ role: "human", content: userInput.trim() }] },
        streamMode: ["updates", "values"],
      });

      let latestEvieMessage: string | null = null;

      for await (const chunk of stream) {
        if (chunk.event === "values") {
          const data = chunk.data as Record<string, unknown>;

          if (data.current_node) {
            const rawMessages = data.messages as Array<Record<string, unknown>> | undefined;
            if (rawMessages) {
              const lastMsg = rawMessages[rawMessages.length - 1];
              const isAi =
                lastMsg?.type === "ai" ||
                (lastMsg as Record<string, unknown>)?.lc_id?.toString().includes("AIMessage");
              const content = lastMsg?.content as string | undefined;
              if (isAi && content && content !== latestEvieMessage) {
                latestEvieMessage = content;
              }
            }

            setEvieState({
              currentNode: (data.current_node as EvieNode) ?? "POLITE_MASK",
              emotion: (data.emotion as string) ?? "relaxed",
              milestones: (data.milestones as string[]) ?? [],
              trustLevel: (data.trust_level as number) ?? 1,
              fearLevel: (data.fear_level as number) ?? 0,
            });
          }
        }
      }

      if (latestEvieMessage) {
        const aiMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: "ai",
          content: latestEvieMessage,
        };
        setMessages((prev) => [...prev, aiMsg]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Connection failed.");
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  const resetThread = useCallback(() => {
    threadIdRef.current = null;
    setMessages([]);
    setEvieState(DEFAULT_STATE);
    setError(null);
  }, []);

  return { messages, evieState, isLoading, error, sendMessage, resetThread };
}
