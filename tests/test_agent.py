from agent import agent_executor
def test_balance():
    result = agent_executor.invoke({"input": "ETH balance of vitalik.eth"})
    assert "ETH" in result["output"]