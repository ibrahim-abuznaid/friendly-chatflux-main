import { toast } from "@/components/ui/use-toast";

const LETTA_BASE_URL = "http://localhost:8283/v1";

export interface LettaAgent {
  id: string;
  name: string;
  user_id: string;
  description?: string;
  created_at?: string;
  tags?: string[];
}

export interface LettaMessage {
  role: "user" | "assistant";
  text: string;
}

interface MemoryBlock {
  type: string;
  value: string;
  label: string;
  category?: string;
  documents?: any[];
  messages?: any[];
}

export const createAgent = async (userId: string): Promise<LettaAgent | null> => {
  try {
    console.log("Creating agent with URL:", LETTA_BASE_URL);
    const agentConfig = {
      name: `agent-${userId}`,
      description: "A test agent for verifying Letta server connection",
      tags: [userId],
      tools: [
        "send_message",
        "conversation_search",
        "archival_memory_search"
      ],
      system: "You are a helpful AI assistant.",
      llm_config: {
        model: "gpt-4o",
        model_endpoint_type: "openai",
        context_window: 182000
      },
      embedding_config: {
        embedding_endpoint_type: "openai",
        embedding_model: "text-embedding-ada-002",
        embedding_dim: 1536
      },
      memory_blocks: [
        {
          type: "core_memory",
          value: "Basic information about the agent and its purpose.",
          label: "identity",
          category: "identity"
        },
        {
          type: "archival_memory",
          value: "Archive memory initialization",
          label: "archive",
          documents: []
        },
        {
          type: "recall_memory",
          value: "Recall memory initialization",
          label: "recall",
          messages: []
        }
      ]
    };

    console.log("Sending agent config:", JSON.stringify(agentConfig, null, 2));

    try {
      const response = await fetch(`${LETTA_BASE_URL}/agents`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(agentConfig)
      });

      const responseText = await response.text();
      console.log('Raw server response:', responseText);

      if (!response.ok) {
        console.error('Server error response:', {
          status: response.status,
          statusText: response.statusText,
          body: responseText
        });
        throw new Error(`Server error: ${response.status} - ${responseText}`);
      }

      let agent;
      try {
        agent = JSON.parse(responseText);
      } catch (parseError) {
        console.error('Failed to parse server response:', parseError);
        throw new Error(`Invalid server response: ${responseText}`);
      }

      if (!agent || !agent.id) {
        console.error('Invalid agent data:', agent);
        throw new Error('Server returned invalid agent data');
      }

      console.log("Agent created successfully:", agent);
      
      toast({
        title: "Success",
        description: "Letta agent created successfully",
      });
      return agent;
    } catch (fetchError) {
      console.error('Fetch error details:', fetchError);
      throw fetchError;
    }
  } catch (error) {
    console.error('Error creating agent:', error);
    console.error('Error stack:', error instanceof Error ? error.stack : 'No stack trace');
    toast({
      title: "Error",
      description: error instanceof Error ? error.message : "Failed to create Letta agent",
      variant: "destructive",
    });
    return null;
  }
};

export const findAgentByUserTag = async (userTag: string): Promise<string | null> => {
  try {
    const response = await fetch(`${LETTA_BASE_URL}/agents`, {
      headers: {
        'Content-Type': 'application/json',
      }
    });

    if (!response.ok) {
      throw new Error('Failed to list agents');
    }

    const agents = await response.json();
    const userAgent = agents.find((agent: LettaAgent) => 
      agent.tags?.includes(userTag)
    );

    return userAgent ? userAgent.id : null;
  } catch (error) {
    console.error('Error finding agent:', error);
    return null;
  }
};

export const sendMessageToAgent = async (
  userId: string,
  message: string
): Promise<string> => {
  try {
    const agentId = await findAgentByUserTag(userId);
    if (!agentId) {
      throw new Error('No agent found for this user');
    }

    const response = await fetch(
      `${LETTA_BASE_URL}/agents/${agentId}/messages`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [{
            role: "user",
            text: message
          }],
          stream_steps: true
        })
      }
    );

    if (!response.ok) {
      throw new Error('Failed to send message');
    }

    // Handle streaming response
    const reader = response.body?.getReader();
    let result = '';

    if (reader) {
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        
        const chunk = new TextDecoder().decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            
            try {
              const parsed = JSON.parse(data);
              if (parsed.message_type === 'function_call' && 
                  parsed.function_call?.name === 'send_message') {
                result = JSON.parse(parsed.function_call.arguments).message;
              }
            } catch (e) {
              console.error('Failed to parse chunk:', e);
            }
          }
        }
      }
    }

    return result || 'Sorry, I could not process that message';
  } catch (error) {
    toast({
      title: "Error",
      description: "Failed to send message to agent",
      variant: "destructive",
    });
    throw error;
  }
}; 