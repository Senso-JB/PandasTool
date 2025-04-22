import sys,re
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import pandas as pd
from tqdm import tqdm


MODEL_NAME = "gemini-2.0-flash"

API_KEY = "AIzaSyBiyCuylzKco47r9XddJNeCYXcmBBYnWOk"

def run(query,file_path="assets/HVAC.csv"):

    df = pd.read_csv(file_path,encoding="ISO-8859-1")


    few_shot_examples = """
Question: How many rows do you have?
Answer: print(len(df))

Question: What are the column names in the DataFrame?
Answer: print(df.columns.tolist())

Question: What is the shape of the DataFrame?
Answer: print(df.shape)

Question: How do I get the first 5 rows of the DataFrame?
Answer: print(df.head())

Question: How can I get summary statistics for all columns?
Answer: print(df.describe())

Question: How do I check for missing values in the DataFrame?
Answer: print(df.isnull().sum())

Question: How do I filter rows where column 'Age' is greater than 30?
Answer: print(df[df['Age'] > 30])

Question: How do I group the data by 'Department' and calculate the average salary?
Answer: print(df.groupby('Department')['Salary'].mean())

Question: What is the maximum value in the 'Capacity' column?
Answer: print(df['Capacity'].max())

Question: Which row has the minimum capacity in the 'Zone' equal to 'East'?
Answer: print(df[df['Zone'] == 'East'].sort_values(by='Capacity').head(1))

Question: How many unique values are there in the 'Category' column?
Answer: print(df['Category'].nunique())

Question: What is the maximum capacity for rows where the 'Type' is 'Storage'?
Answer: print(df[df['Type'] == 'Storage']['Capacity'].max())

Question: How do I find rows where the 'Product' column contains the exact word "Steel", case-sensitive?
Answer: print(df[df['Product'].str.contains('Steel', case=True, na=False)])

Question: How do I filter rows where the 'Type' column exactly matches "Storage", case-sensitive?
Answer: print(df[df['Type'].str.match('Storage', case=True, na=False)])

Question: How do I check if 'Zone' contains an exact case match for "East"?
Answer: print(df[df['Zone'].str.contains('East', case=True, na=False)])
"""


    prompt_template = """
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
        
        {tool_names}
        
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
        
        Question: {input}"""
    prompt = PromptTemplate(
        template=prompt_template
    )

    custom_prompt = prompt.format(tool_names="python_repl_ast",input=query,examples=few_shot_examples)

    # print(custom_prompt)

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        temperature=0,
        max_tokens=1000,
        top_k=10
    )
    
    # logging.info(f"IBM MODEL : {IBM_MODEL}")
    # llm = WatsonxLLM(
    #     model_id=IBM_MODEL,
    #     project_id=WATSONX_PROJECT_ID,
    #     apikey=WATSONX_API_KEY,
    #     url=SERVER_URL,
    #     params=WASTSONX_PARAMS
    # )

    # Create a custom agent
    pandas_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=False,
        prefix=custom_prompt,
        allow_dangerous_code=True,
        agent_executor_kwargs={'handle_parsing_errors':True},
        max_iterations=15,
    )

    response = pandas_agent.invoke(query)
    return response['output']

if __name__ == "__main__":
    import time
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
    i  = 0
    while i < len(queries):
        try:
            query = queries[i]
            print("-"*dash)
            print(query)
            print("-"*dash)
            response = run(query=query)
            chat += f"\nQuestion{i+1} : {query} \n"
            print("-"*dash)
            print(response)
            chat += f"Answer :\n{response} \n"
            print("-"*dash)
            i+=1
        except Exception as e:
            error_str = str(e)
            match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
            if match:
                delay_seconds = int(match.group(1))
                for t in tqdm(range(delay_seconds),desc="Please wait",unit="sec"):
                    time.sleep(0.5)
            continue

    with open("v2.txt","w") as file:
        file.write(chat)

    # while True:
    #     try:
    #         user = input("You :")
    #         if user in ('exit','quit'):
    #             break
    #         bot = run(query=user)
    #         print("Bot :\n",bot)
    #     except Exception as e:
    #         print(e)