import { ChatOpenAI } from "@langchain/openai";
import { NPCResponseSchema } from "./npc.js";

const baseURL = process.env.LMSTUDIO_BASE_URL ?? "http://localhost:1234/v1";
const apiKey = process.env.LMSTUDIO_API_KEY ?? "lm-studio";
const modelName = process.env.LMSTUDIO_MODEL ?? "gemma-4-e4b";

export const llm = new ChatOpenAI({
  model: modelName,
  apiKey,
  temperature: 0.7,
  configuration: { baseURL },
});

export const structuredLlm = llm.withStructuredOutput(NPCResponseSchema, {
  name: "NPCResponse",
});
