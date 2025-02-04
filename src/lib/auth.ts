import { toast } from "@/components/ui/use-toast";

export interface User {
  id: string;
  email: string;
}

// Simulated user storage (replace with actual backend later)
const users: User[] = [];

export const signup = async (email: string, password: string): Promise<User | null> => {
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

  // Create new user
  const user: User = {
    id: Math.random().toString(36).substr(2, 9),
    email,
  };
  users.push(user);
  localStorage.setItem("user", JSON.stringify(user));
  return user;
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