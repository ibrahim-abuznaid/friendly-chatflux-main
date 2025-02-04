// server.js
import express from "express";
import OpenAI from "openai";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(express.json());
app.use(cors()); // Enable CORS so your frontend can talk to this backend

const OPENAI_API_KEY = process.env.VITE_OPENAI_API_KEY;
const ASSISTANT_ID = process.env.VITE_ASSISTANT_ID; // Your pre‑created assistant’s ID

if (!OPENAI_API_KEY || !ASSISTANT_ID) {
  console.error("Missing API key or assistant ID");
  process.exit(1);
}

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
  // Do not need dangerouslyAllowBrowser here because this code runs on the server
});

app.post("/api/send-message", async (req, res) => {
  const { message } = req.body;
  try {
    // 1. Create a new thread.
    const thread = await openai.beta.threads.create();
    const threadId = thread.id;

    // 2. Post the user message to the thread.
    await openai.beta.threads.messages.create(threadId, {
      role: "user",
      content: message,
    });

    // 3. Create a run for the thread.
    const run = await openai.beta.threads.runs.create(threadId, {
      assistant_id: ASSISTANT_ID,
      instructions: "Please respond to the user message.",
      response_format: { type: "text" },
    });

    // 4. Poll for run completion.
    let currentRun = run;
    while (
      currentRun.status !== "completed" &&
      currentRun.status !== "failed"
    ) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      currentRun = await openai.beta.threads.runs.retrieve(threadId, run.id);
    }
    if (currentRun.status !== "completed") {
      throw new Error("Assistant run did not succeed");
    }

    // 5. Retrieve all messages from the thread.
    const messagesResponse = await openai.beta.threads.messages.list(threadId);
    const messages = messagesResponse.data.map((msg) => ({
      role: msg.role,
      content: msg.content.reduce((acc, item) => {
        if ("text" in item) return acc + item.text.value;
        return acc;
      }, ""),
    }));

    const assistantMessage = messages.reverse().find(
      (msg) => msg.role === "assistant"
    );
    if (!assistantMessage) {
      throw new Error("No assistant reply found");
    }

    res.json({ reply: assistantMessage.content });
  } catch (error) {
    console.error("Error in /api/send-message:", error);
    res.status(500).json({
      error:
        error instanceof Error ? error.message : "Failed to send message",
    });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
