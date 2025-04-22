import pandas as pd
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_ibm.chat_models import ChatWatsonx
from langchain_core.prompts import PromptTemplate

class CleanPythonREPLTool(PythonAstREPLTool):
    def run(self, code: str) -> str:
        result = super().run(code)
        return result.strip().strip("```python").strip("```").strip()
    

import re

def safe_execute_python_with_action_input(raw_input: str) -> str:
    print(globals())
    try:
        # Remove leading and trailing whitespaces and clean input
        code = raw_input.strip()

        # Prepare local variables (ensure the DataFrame `df` is available)
        local_vars = {"df": df}

        try:
            result = eval(code, {}, globals())
        except SyntaxError:
            exec(code, {}, globals())
            result = local_vars.get("result", "Executed. No explicit result.")

        return str(result)

    except Exception as e:
        return f"Error: {str(e)}"

    

# --- Config ---
SERVER_URL = "https://au-syd.ml.cloud.ibm.com"
WATSONX_PROJECT_ID = "853fb52b-1dec-44f4-a11f-29bf0229f522"
WATSONX_API_KEY = "k6c3ma08YuR7UHw9ilLakBoqwDywXcihZUxUt_uJTEWG"

IBM_MODEL = "mistralai/mistral-large"

WASTSONX_PARAMS = {
    "temperature": 0,
    "max_new_tokens": 500,
    "min_new_tokens": 10,
    'top_k': 3,
}

# --- Load Data ---
df = pd.read_csv("assets/HVAC.csv", encoding="ISO-8859-1")

globals()['df'] = df

# --- LLM Setup ---
llm = ChatWatsonx(
    model_id=IBM_MODEL,
    project_id=WATSONX_PROJECT_ID,
    apikey=WATSONX_API_KEY,
    url=SERVER_URL,
    params=WASTSONX_PARAMS
)

# --- Tools ---
python_tool = CleanPythonREPLTool(locals={"df": df})

execution_tool = Tool(
    name="pandas_code_executor",
    func=safe_execute_python_with_action_input,
    description="Executes Python code on a pandas DataFrame `df`. Input should be valid Python using the `df` variable."
)
    

tools = [
    Tool(
        name="python_repl_ast",
        func=python_tool.run,
        description="Executes Python code to answer questions about the DataFrame `df`."
    ),
    execution_tool
]

# --- Prompt ---
prompt_template = """
You are a Python data analyst. You are working with a pandas DataFrame called `df`.

You can use the following tool:
[{tool_names}]

Use the following format:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, written as plain Python code (not in backticks or markdown formatting)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
"""

query = "How many rows do you have?"

prompt = PromptTemplate.from_template(prompt_template).format(tool_names="python_repl_ast,pandas_code_executor",input=query)

# --- Agent ---
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prefix": prompt},
    max_iterations=10,
    handle_parsing_errors=True,
)

# --- Run ---

response = agent_executor.run(query)
print("\n✅ Final Response:", response)
