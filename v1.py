from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonAstREPLTool
import pandas as pd
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import time,re
from tqdm import tqdm

MODEL_NAME = "gemini-2.0-flash"

API_KEY = "AIzaSyBiyCuylzKco47r9XddJNeCYXcmBBYnWOk"


class DataFrameQuerySystem:
    def __init__(self):
        """Initialize the DataFrameQuerySystem with a language model."""
        self.model = ChatGoogleGenerativeAI(model=MODEL_NAME,temperature=0,api_key=API_KEY)
        
        self.dataframes = {}
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()
        
    def _create_tools(self) -> list:
        """Create the tools for the agent."""
        # Create a dictionary with loaded dataframes that will be accessible in the REPL
        globals_dict = {
            "pd": pd,
            "df_dict": self.dataframes
        }
        
        return [
            PythonAstREPLTool(
                name="python_repl",
                description="""Execute Python code for data analysis. 
                The dataframes are available as variables in the df_dict dictionary.
                For example, if 'sales' is a loaded dataframe, access it with df_dict['sales'].
                You can create visualizations using matplotlib (plt).
                Use pandas (pd) for data manipulation.""",
                globals=globals_dict
            ),
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create a ReAct agent for dataframe querying."""
        react_prompt = PromptTemplate.from_template("""
        You are a data analysis assistant using Python and pandas to analyze dataframes.
        
        To answer user queries, follow these steps:
        1. Understand what the user is asking about the data
        2. Plan your approach using the available tools
        3. If needed, explore the data structure before proceeding
        4. Execute Python code to answer the query
        5. Interpret the results
        6. Present a clear, concise answer
        
        Available dataframes are in the df_dict dictionary.
        
        You have access to the following tools:
        
        {tools}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        When working with the data:
        - First check the dataframe structure if you're not familiar with it
        - Use efficient pandas operations
        - For complex queries, break down your analysis into steps
        - Focus on answering exactly what was asked
        - Format results nicely and explain what they mean
        - If you're creating visualizations, make sure to include plt.tight_layout() and plt.show()
        
        Begin!
        
        Question: {input}
        {agent_scratchpad}
        """)
        
        agent = create_react_agent(self.model, self.tools, react_prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True
        )
    
    def _view_dataframe_info(self) -> str:
        """Return information about the loaded dataframes."""
        if not self.dataframes:
            return "No dataframes are currently loaded."
        
        result = []
        for name, df in self.dataframes.items():
            result.append(f"Dataframe: {name}")
            result.append(f"Shape: {df.shape}")
            result.append("Columns:")
            for col in df.columns:
                dtype = df[col].dtype
                result.append(f"  - {col} ({dtype})")
            result.append("\nSample data:")
            result.append(str(df.head(3)))
            result.append("\n" + "-"*50 + "\n")
        
        return "\n".join(result)
    
    def load_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """Load a dataframe into the system."""
        self.dataframes[name] = df
        print(f"Dataframe '{name}' loaded successfully.")
        
        # We need to update the tools when new dataframes are loaded
        # This ensures the PythonAstREPLTool has access to the updated dataframes
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()
    
    def query(self, user_query: str) -> str:
        """Process a user query about the loaded dataframes."""
        if not self.dataframes:
            return "Please load at least one dataframe before querying."
        result = self.agent_executor.invoke({"input": user_query})
        return result["output"]

# Example usage
if __name__ == "__main__":
    # Create the query system
    query_system = DataFrameQuerySystem()
    
    # Load example dataframes
    df = pd.read_csv("assets/HVAC.csv", encoding="ISO-8859-1")
    
    # Load the dataframes into the system
    query_system.load_dataframe(df, 'hvac')
    
    # Example queries
    queries = [
        "How many rows are there ?",
        "what is the maximum capacity of primary chilled water pump ?",
        "what is the maximum power of exahust air",
        "List the unqiue application names",
        "Which model numbers are used for equipment in PHASE-2?",
        "how manu applications are available ?",
        "Which application consumes higher energy ?",
        "How many types of chillers found ?",
        "how many types of equipment tag are available ?",
        "List all equipment that use a 'DIRECT FEEDER' starter type.",
        "How many applications falls under the Chiller Equipment tag ?",
        "What is the total number of chillers installed in the CHILLER PLANT ROOM?",
        "Which equipment type has the highest power input (in kW)?",
        "Which equipment serves the 'ENTIRE PLANT' area, and what are their applications?",
        "How many different equipment types are there?",
    ]
    
    dash = len(max(queries,key=len))
    chat = ""

    i = 0
    while i < len(queries):
        try:
            question = queries[i]
            print("-"*dash)
            print(question)
            print("-"*dash)
            response = query_system.query(user_query=question)
            chat += f"\nQuestion{i+1} : {question} \n"
            print("-"*dash)
            print(response)
            chat += f"Answer :\n{response} \n"
            print("-"*dash)
            i+=1
            continue
        except Exception as e:
            error_str = str(e)
            match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
            if match:
                delay_seconds = int(match.group(1))
                for t in tqdm(range(delay_seconds),desc="Please wait",unit="sec"):
                    time.sleep(0.5)
            continue
    
    with open("v1.txt","a") as file:
        file.write(chat)