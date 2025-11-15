import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from web3 import Web3
import requests
from graphql_request import GraphQLRequest  # pip install graphql-request (wait, use requests for GraphQL)

load_dotenv()

# Setup LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# DeFi Docs for RAG (pre-loaded vector store)
def setup_rag_docs():
    docs = [
        Document(page_content="Uniswap V3: Use subgraph for pool prices. Query: { pool { token0Price } }"),
        Document(page_content="1inch API: POST /v5.0/1/quote for swap quotes. Endpoint: https://api.1inch.io/v5.0/1/quote"),
        Document(page_content="web3.py: w3.eth.get_balance(address) for ETH balance."),
        Document(page_content="ethers.js equivalent: ethers.provider.getBalance(address)"),
    ]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

rag_docs = setup_rag_docs()

# Custom Tools
@tool
def get_eth_balance(wallet_address: str) -> str:
    """Get ETH balance for a wallet using web3.py."""
    w3 = Web3(Web3.HTTPProvider(os.getenv("INFURA_URL")))
    if not w3.is_connected():
        return "Error: Failed to connect to Ethereum."
    balance_wei = w3.eth.get_balance(wallet_address)
    balance_eth = w3.from_wei(balance_wei, 'ether')
    return f"ETH Balance for {wallet_address}: {balance_eth:.4f} ETH"

@tool
def uniswap_price(token_address: str) -> str:
    """Get token price from Uniswap V3 subgraph (ETH pair)."""
    query = """
    query($token: String!) {
      token(id: $token) {
        derivedETH
      }
    }
    """
    variables = {"token": token_address.lower()}
    response = requests.post(
        "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
        json={"query": query, "variables": variables}
    )
    data = response.json()
    if "errors" in data:
        return f"Error: {data['errors']}"
    price_eth = data['data']['token']['derivedETH']
    return f"Token {token_address} price in ETH: {price_eth}"

@tool
def inch_quote(from_token: str, to_token: str, amount: str) -> str:
    """Get swap quote from 1inch API."""
    url = "https://api.1inch.io/v5.0/1/quote"
    params = {
        "fromTokenAddress": from_token,
        "toTokenAddress": to_token,
        "amount": amount  # Amount in wei
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "error" in data:
        return f"Error: {data['error']}"
    return f"Quote: {data['toTokenAmount']} {to_token} for {amount} {from_token}"

@tool
def generate_ethers_js_code(action: str) -> str:
    """Generate ethers.js code snippet for a DeFi action (e.g., 'swap on Uniswap')."""
    prompt = f"Generate concise ethers.js code for: {action}. Use provider and wallet."
    return llm.invoke(prompt).content  # LLM generates JS code

# Agent Setup
tools = [get_eth_balance, uniswap_price, inch_quote, generate_ethers_js_code]

prompt = PromptTemplate.from_template(
    """You are a DeFi AI Agent. Use tools to answer queries about blockchain, prices, swaps, or generate JS code.
    Use web3.py for Python actions, ethers.js for JS generation.
    Always reason step-by-step.

{input}
{agent_scratchpad}"""
)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# RAG Retrieval (integrated into agent via custom tool if needed)
@tool
def retrieve_defi_docs(query: str) -> str:
    """Retrieve DeFi docs via RAG."""
    docs = rag_docs.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in docs])

tools.append(retrieve_defi_docs)  # Add to tools

# Example Usage
if __name__ == "__main__":
    query = "What's the ETH balance of 0x742d35Cc6634C0532925a3b8D7fE6D8cC6a7aB6D? Then get a quote for 1 ETH to USDC on 1inch."
    result = agent_executor.invoke({"input": query})
    print("Agent Response:", result["output"])

    # JS Generation Example
    js_query = "Generate ethers.js code to check Uniswap ETH price."
    js_result = agent_executor.invoke({"input": js_query})
    print("Generated JS Code:", js_result["output"])
    