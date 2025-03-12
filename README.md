

Attempt to make an agent that is capable of training an XGBoost model with titanic dataset.
The agent cannot directly access python interpreter or terminal to run shell commands.
I provided tool for it but difficult to make it use. 

titanic_agent.py : The triage_agent will handoff to terminal agent and the terminal agent print out pip install commands and the runner is terminated.

titanic_agent_astools : print out the xgboost training code and terminate.

titanic_agent_whileture : print out the xgboost training code and then stuck in an infinite loop of triage_agent saying "let me know if you need help" kind of message.


References:
[Official Github Example](https://github.com/openai/openai-agents-python/tree/main/examples)


