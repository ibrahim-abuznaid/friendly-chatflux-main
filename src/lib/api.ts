import { toast } from "@/components/ui/use-toast";

export interface ChatResponse {
  reply: string;
  context: {
    original_query: string;
    refined_query: string;
    retrieved_documents: Array<{
      page_content: string;
      metadata: Record<string, any>;
    }>;
  };
}

export const sendMessage = async (message: string): Promise<ChatResponse> => {
  try {
    const response = await fetch("http://localhost:3001/api/send-message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Failed to send message");
    }

    return await response.json();
  } catch (error) {
    toast({
      title: "Error",
      description: error instanceof Error ? error.message : "Failed to send message",
      variant: "destructive",
    });
    throw error;
  }
}; 