import { toast } from "@/components/ui/use-toast";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export const sendMessage = async (message: string): Promise<string> => {
  try {
    // This is where we'll integrate with OpenAI's API
    // For now, return a mock response
    await new Promise((resolve) => setTimeout(resolve, 1000));
    return "This is a mock response. Please connect to OpenAI API for real responses.";
  } catch (error) {
    toast({
      title: "Error",
      description: "Failed to send message",
      variant: "destructive",
    });
    throw error;
  }
};