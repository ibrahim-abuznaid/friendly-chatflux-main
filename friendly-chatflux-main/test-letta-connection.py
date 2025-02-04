import requests
import json

# Configuration
LETTA_BASE_URL = "http://localhost:8283/v1"

def test_list_agents():
    """Test listing all agents"""
    print("\n🔄 Testing Agent Listing...")
    print(f"📍 Connecting to: {LETTA_BASE_URL}")
    
    try:
        response = requests.get(
            f"{LETTA_BASE_URL}/agents",
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            agents = response.json()
            print("\n✅ Successfully retrieved agents!")
            print("\nFound agents:")
            print("-" * 50)
            
            if not agents:
                print("No agents found.")
            else:
                for agent in agents:
                    print(f"Agent Name: {agent.get('name', 'N/A')}")
                    print(f"Agent ID: {agent.get('id', 'N/A')}")
                    print(f"Description: {agent.get('description', 'N/A')}")
                    print(f"User ID: {agent.get('user_id', 'N/A')}")
                    print("-" * 30)
            
            print("-" * 50)
            return agents
        else:
            print(f"\n❌ Failed to list agents. Status code: {response.status_code}")
            print(f"Error: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Could not connect to Letta server")
        print(f"   Make sure the Letta server is running on {LETTA_BASE_URL}")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return None

if __name__ == "__main__":
    test_list_agents() 