import { toast } from "@/components/ui/use-toast";
import { createAgent } from "./letta";
import { v4 as uuidv4 } from 'uuid';

export interface User {
  id: string;
  email: string;
  agentId?: string;
}

// Simulated user storage (replace with actual backend later)
const users: User[] = [];

// Generate a unique numeric ID
function generateUserId(): string {
  // Generate a 12-digit number by taking first 12 digits of a UUID without dashes
  const uuid = uuidv4().replace(/-/g, '');
  const numericId = uuid.slice(0, 12);
  
  // Ensure uniqueness (in a real app, this would be handled by the database)
  if (users.some(user => user.id === numericId)) {
    return generateUserId(); // Recursively try again if ID exists
  }
  
  return numericId;
}

export const signup = async (email: string, password: string): Promise<User | null> => {
  try {
    // Simple validation
    if (!email || !password) {
      toast({
        title: "Error", 
        description: "Please provide both email and password",
        variant: "destructive",
      });
      return null;
    }

    // Check if user exists
    if (users.find((u) => u.email === email)) {
      toast({
        title: "Error",
        description: "User already exists",
        variant: "destructive",
      });
      return null;
    }

    // Create new user with unique ID
    const userId = generateUserId();
    const user: User = {
      id: userId,
      email,
    };

    console.log("Starting Letta agent creation for user:", user);

    // Create Letta agent for the user
    try {
      console.log("Creating agent for user:", user.id);
      const agent = await createAgent(user.id);
      
      if (!agent) {
        console.error("Agent creation returned null");
        throw new Error("Failed to create Letta agent - null response");
      }

      console.log("Agent created successfully:", agent);
      user.agentId = agent.id;
      
    } catch (error) {
      console.error("Letta agent creation error:", error);
      console.error("Error details:", {
        name: error instanceof Error ? error.name : 'Unknown',
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : 'No stack trace'
      });
      
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to create AI assistant. Please try again.",
        variant: "destructive",
      });
      return null;
    }

    users.push(user);
    localStorage.setItem("user", JSON.stringify(user));
    
    toast({
      title: "Success",
      description: "Account created successfully!",
    });
    
    return user;
  } catch (error) {
    console.error("Signup error:", error);
    toast({
      title: "Error",
      description: "Failed to create account. Please try again.",
      variant: "destructive",
    });
    return null;
  }
};

export const login = async (email: string, password: string): Promise<User | null> => {
  const user = users.find((u) => u.email === email);
  if (!user) {
    toast({
      title: "Error",
      description: "Invalid credentials",
      variant: "destructive",
    });
    return null;
  }
  localStorage.setItem("user", JSON.stringify(user));
  return user;
};

export const logout = () => {
  localStorage.removeItem("user");
};

export const getCurrentUser = (): User | null => {
  const userStr = localStorage.getItem("user");
  return userStr ? JSON.parse(userStr) : null;
};