import requests
import json

# Configuration
LETTA_BASE_URL = "http://localhost:8283/v1"

def test_create_agent():
    """Test creating a new Letta agent"""
    print("\n🔄 Testing Agent Creation...")
    print(f"📍 Connecting to: {LETTA_BASE_URL}")
    
    # Agent configuration
    agent_config = {
        "name": "test-agent5",
        "description": "A test agent for verifying Letta server connection",
        "user_id": "test-user-123",
        "tools": [
            "send_message",
            "conversation_search",
            "archival_memory_search"
        ],
        "system": "You are a helpful AI assistant.",
        "llm_config": {
            "model": "gpt-4o",
            "model_endpoint_type": "openai",
            "context_window": 182000
        },
        "embedding_config": {
            "embedding_endpoint_type": "openai",
            "embedding_model": "text-embedding-ada-002",
            "embedding_dim": 1536
        },
        "memory_blocks": [
            {
                "type": "core_memory",
                "value": "Basic information about the agent and its purpose.",
                "label": "identity",
                "category": "identity"
            },
            {
                "type": "archival_memory",
                "value": "Archive memory initialization",
                "label": "archive",
                "documents": []
            },
            {
                "type": "recall_memory",
                "value": "Recall memory initialization",
                "label": "recall",
                "messages": []
            }
        ]
    }
    
    try:
        print("\nAttempting to create agent with configuration:")
        print("-" * 50)
        print(f"Name: {agent_config['name']}")
        print(f"Description: {agent_config['description']}")
        print(f"User ID: {agent_config['user_id']}")
        print(f"Tools: {', '.join(agent_config['tools'])}")
        print(f"Memory Blocks: {[block['type'] for block in agent_config['memory_blocks']]}")
        print("-" * 50)
        
        response = requests.post(
            f"{LETTA_BASE_URL}/agents",
            headers={"Content-Type": "application/json"},
            json=agent_config
        )
        
        if response.status_code == 200:
            agent_data = response.json()
            print("\n✅ Successfully created agent!")
            print("\nAgent Details:")
            print("-" * 50)
            print(f"Agent Name: {agent_data.get('name', 'N/A')}")
            print(f"Agent ID: {agent_data.get('id', 'N/A')}")
            print(f"Description: {agent_data.get('description', 'N/A')}")
            print(f"Created At: {agent_data.get('created_at', 'N/A')}")
            print("-" * 50)
            return agent_data
        else:
            print(f"\n❌ Failed to create agent. Status code: {response.status_code}")
            print(f"Error: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Could not connect to Letta server")
        print(f"   Make sure the Letta server is running on {LETTA_BASE_URL}")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return None

def verify_agent_creation(agent_data):
    """Verify the created agent by listing all agents"""
    if not agent_data:
        return False
        
    print("\n🔄 Verifying agent creation...")
    
    try:
        response = requests.get(
            f"{LETTA_BASE_URL}/agents",
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            agents = response.json()
            agent_exists = any(
                agent.get('id') == agent_data.get('id') 
                for agent in agents
            )
            
            if agent_exists:
                print("✅ Agent successfully verified in the agents list!")
                return True
            else:
                print("❌ Created agent not found in the agents list")
                return False
        else:
            print(f"❌ Failed to verify agent. Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to verify agent: {str(e)}")
        return False

if __name__ == "__main__":
    # Create the agent
    created_agent = test_create_agent()
    
    # Verify the creation
    if created_agent:
        verify_agent_creation(created_agent) 