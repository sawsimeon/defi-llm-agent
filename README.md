# DeFi LLM Agent: LangChain + web3.py + Ethers.js Code Gen

An AI agent for DeFi queries: balances, prices, swaps via web3.py, with LLM-generated ethers.js for JS frontends.

## Quick Start
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill keys.
3. `python agent.py`

## Tools
- **web3.py**: On-chain queries (balances).
- **Uniswap/1inch APIs**: Prices/swaps.
- **LangChain ReAct Agent**: Reasoning + tool calls.
- **RAG (FAISS)**: DeFi docs retrieval.
- **Ethers.js Gen**: LLM outputs JS code snippets.

## Examples
- "Swap 1 ETH to DAI quote?": Calls 1inch API.
- "Generate JS for Uniswap approval": Outputs ethers.js code.

## License
MIT.