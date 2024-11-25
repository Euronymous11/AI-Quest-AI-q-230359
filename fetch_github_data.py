import requests
import json

# GitHub token for authentication
TOKEN = "My_github_token_is_placed_here"
HEADERS = {"Authorization": f"token {TOKEN}"}

# Updated Topics List
search_topics = [
    "AI", "machine-learning", "transformers", "gpt", "rag-model", "openai-api", 
    "knowledge-retrieval", "question-answering", "chatbot-development", "mlops", 
    "model-deployment", "information-retrieval", "elastic-search", "vector-databases", 
    "langchain", "pinecone", "huggingface", "pytorch", "tensorflow", 
    "developer-tools", "community-engagement", "knowledge-management", "neural-networks", 
    "enterprise-search", "internal-wikis", "document-parsing", "semantic-search", 
    "text-summarization", "nlp-tasks", "contextual-search", "stackoverflow-api", "github-discussions"
]

# Function to fetch repositories
def fetch_repositories(topic, max_pages=10):
    repositories = []
    for page in range(1, max_pages + 1):
        print(f"Fetching repositories for topic: {topic}, Page {page}")
        url = f"https://api.github.com/search/repositories?q=topic:{topic}&sort=stars&order=desc&page={page}&per_page=100"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            repositories.extend(data.get("items", []))
        else:
            print(f"Error fetching repositories for topic {topic}, Page {page}: {response.status_code}")
            break
    return repositories

# Function to fetch discussions or fallback to issues
def fetch_data(repo_owner, repo_name):
    discussions_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/discussions"
    issues_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"

    # Try fetching discussions
    response = requests.get(discussions_url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 410:  # Discussions not available, fallback to issues
        print(f"Discussions unavailable for {repo_owner}/{repo_name}. Fetching issues instead.")
        response = requests.get(issues_url, headers=HEADERS)
        if response.status_code == 200:
            return response.json()
    else:
        print(f"Error fetching data for {repo_owner}/{repo_name}: {response.status_code}")
    return None

# Main function
def main():
    all_data = {}
    for topic in search_topics:
        repositories = fetch_repositories(topic, max_pages=10)
        for repo in repositories:
            repo_owner = repo["owner"]["login"]
            repo_name = repo["name"]
            print(f"Fetching data for {repo_owner}/{repo_name}")
            repo_data = fetch_data(repo_owner, repo_name)
            if repo_data:
                all_data[f"{repo_owner}/{repo_name}"] = repo_data

    # Save data to JSON
    with open("github_data.json", "w") as json_file:
        json.dump(all_data, json_file, indent=4)
    print("Data saved to github_data.json")

if __name__ == "__main__":
    main()
