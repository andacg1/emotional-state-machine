import { z } from "zod";

export const NPCEmotion = z.enum([
  "relaxed",
  "nervous",
  "panicking",
  "angry",
  "upset",
  "depressed",
  "defensive",
]);
export type NPCEmotion = z.infer<typeof NPCEmotion>;

export const NPCResponseSchema = z.object({
  message: z.string(),
  emotion: NPCEmotion,
  topic: z.string(),
  summary: z.string(),
  knowledge: z.array(z.string()).nullable().default(null),
});

export type NPCResponse = z.infer<typeof NPCResponseSchema>;
