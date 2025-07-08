import json
import mysql.connector
import wikipedia
from typing import Dict, List, Any, Optional
import openai
from dotenv import load_dotenv
import os
import re

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_GPT_MODEL = os.getenv("OPENAI_GPT_MODEL", "gpt-4o")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "appuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "example")
DB_NAME = os.getenv("DB_NAME", "appdb")

# First, we define how the LLM should understand our tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": """Execute a MySQL SELECT query and return the results.
                            Only SELECT queries are allowed for security reasons.
                            Returns data in JSON format.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": """The SQL SELECT query to execute.
                                        Must start with SELECT and should not
                                        contain any data modification commands."""
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": """Search Wikipedia and return a concise summary.
                            Returns the first three sentences of the most
                            relevant article.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic to search for on Wikipedia"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

def get_table_schema():
    conn = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    schema_info = []
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES")
        tables = [row[0] for row in cur.fetchall()]
        for table in tables:
            cur.execute(f"SHOW COLUMNS FROM {table}")
            columns = [row[0] for row in cur.fetchall()]
            schema_info.append(f"表 {table} 字段: {', '.join(columns)}")
    return '\n'.join(schema_info)

# Now implement the actual tool functions
def query_database(query: str) -> str:
    """
    Execute a MySQL query and return results as a JSON string.
    Includes safety checks and error handling.
    """
    # Security check: Only allow SELECT queries
    if not query.lower().strip().startswith('select'):
        return json.dumps({
            "error": "Only SELECT queries are allowed for security reasons."
        })

    try:
        # You should use environment variables for these in production
        conn = mysql.connector.connect(
            host="localhost",
            port=3306,
            user="appuser",
            password="example",
            database="appdb"
        )

        # 获取所有表名
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            table_rows = cur.fetchall()
            table_names = [row[0] for row in table_rows]

        # 简单模糊匹配：将 SQL 语句中的表名与实际表名做近似替换（如复数转单数）
        def match_table(word):
            # 完全匹配
            if word in table_names:
                return word
            # 复数转单数（只允许 tasks->task，不允许 task->tasks）
            if word.endswith('s') and word[:-1] in table_names:
                return word[:-1]
            # 忽略大小写匹配
            for t in table_names:
                if word.lower() == t.lower():
                    return t
            return word  # 不变

        # 用正则查找 SQL 语句中的表名并替换
        def table_replacer(match):
            # 保证替换后有空格
            return f"{match.group(1)} {match_table(match.group(2))}{match.group(3)}"

        # 只替换 FROM/JOIN 后的表名，允许 FROMtasks/JOINtasks 也能被识别
        query = re.sub(r'(from|join)\s*([a-zA-Z0-9_]+)(\b)', table_replacer, query, flags=re.IGNORECASE)

        print(f"Executing query: {query}")

        with conn.cursor() as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            results = []
            for row in cur.fetchall():
                results.append(dict(zip(columns, row)))
            return json.dumps({
                "success": True,
                "data": str(results),
                "row_count": len(results)
            })

    except mysql.connector.Error as e:
        return json.dumps({
            "error": f"Database error: {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "error": f"Unexpected error: {str(e)}"
        })
    finally:
        if 'conn' in locals():
            conn.close()

def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia and return a concise summary.
    Handles disambiguation and missing pages gracefully.
    """
    try:
        # Try to get the most relevant page summary
        summary = wikipedia.summary(
            query,
            sentences=3,
            auto_suggest=True,
            redirect=True
        )

        return json.dumps({
            "success": True,
            "summary": summary,
            "url": wikipedia.page(query).url
        })

    except wikipedia.DisambiguationError as e:
        # Handle multiple matching pages
        return json.dumps({
            "error": "Disambiguation error",
            "options": e.options[:5],  # List first 5 options
            "message": "Topic is ambiguous. Please be more specific."
        })
    except wikipedia.PageError:
        return json.dumps({
            "error": "Page not found",
            "message": f"No Wikipedia article found for: {query}"
        })
    except Exception as e:
        return json.dumps({
            "error": "Unexpected error",
            "message": str(e)
        })

def execute_sql(sql: str) -> str:
    """
    Execute an arbitrary SQL statement (for admin/debug use only).
    Returns a JSON string with success or error message.
    """
    try:
        conn = mysql.connector.connect(
            host="localhost",
            port=3306,
            user="appuser",
            password="example",
            database="appdb"
        )
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
        return json.dumps({
            "success": True,
            "message": f"Executed: {sql}"
        })
    except mysql.connector.Error as e:
        return json.dumps({
            "error": f"Database error: {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "error": f"Unexpected error: {str(e)}"
        })
    finally:
        if 'conn' in locals():
            conn.close()

class Agent:
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize an AI Agent with optional system prompt.

        Args:
            system_prompt: Initial instructions for the AI
        """
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.chatanywhere.tech")

        # Initialize conversation history
        self.messages = []

        # Set up system prompt if provided, otherwise use default
        schema_info = get_table_schema()
        print(f"\nsschema_info: {schema_info}")
        default_prompt = f"""You are a helpful AI assistant with access to a database and Wikipedia. Follow these rules:
        1. When asked about data, always check the database first. Database schema as followings
        {schema_info}
        2. For general knowledge questions, use Wikipedia
        3. If you're unsure about data, query the database to verify
        4. Always mention your source of information
        5. If a tool returns an error, explain the error to the user clearly
        """
        
        self.messages.append({
            "role": "system",
            "content": system_prompt or default_prompt
        })

        print(f"\nself.messages: {self.messages}")

    def execute_tool(self, tool_call: Any) -> str:
        """
        Execute a tool based on the LLM's decision.

        Args:
            tool_call: The function call object from OpenAI's API

        Returns:
            str: JSON-formatted result of the tool execution
        """
        try:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Execute the appropriate tool. Add more here as needed.
            if function_name == "query_database":
                result = query_database(function_args["query"])
            elif function_name == "search_wikipedia":
                result = search_wikipedia(function_args["query"])
            else:
                result = json.dumps({
                    "error": f"Unknown tool: {function_name}"
                })

            return result

        except json.JSONDecodeError:
            return json.dumps({
                "error": "Failed to parse tool arguments"
            })
        except Exception as e:
            return json.dumps({
                "error": f"Tool execution failed: {str(e)}"
            })

    def process_query(self, user_input: str) -> str:
        """
        Process a user query through the AI agent.

        Args:
            user_input: The user's question or command

        Returns:
            str: The agent's response
        """
        # Add user input to conversation history
        self.messages.append({
            "role": "user",
            "content": user_input
        })

        try:
            max_iterations = 5
            current_iteration = 0

            while current_iteration < max_iterations:  # Limit to 5 iterations
                current_iteration += 1
                completion = self.client.chat.completions.create(
                    model=OPENAI_GPT_MODEL,
                    messages=self.messages,
                    tools=tools,  # Global tools list from Step 1
                    tool_choice="auto"  # Let the model decide when to use tools
                )
                response_message = completion.choices[0].message

                print(f"\n@@@@@response_message: {response_message}")

                if not response_message.tool_calls:
                    self.messages.append(response_message)
                    return response_message.content
                self.messages.append(response_message)
                for tool_call in response_message.tool_calls:
                    try:
                        print("Tool call:", tool_call)
                        result = self.execute_tool(tool_call)
                        print("Tool executed......")
                    except Exception as e:
                        print("Execution failed......")
                        result = json.dumps({
                            "error": f"Tool execution failed: {str(e)}"
                        })
                    print(f"Tool result custom: {result}")
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
                    print("Messages:", self.messages)

            # If we've reached max iterations, return a message indicating this
            max_iterations_message = {
                "role": "assistant",
                "content": "I've reached the maximum number of tool calls (5) without finding a complete answer. Here's what I know so far: " + response_message.content
            }
            self.messages.append(max_iterations_message)
            return max_iterations_message["content"]

        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            self.messages.append({
                "role": "assistant",
                "content": error_message
            })
            return error_message

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.

        Returns:
            List[Dict[str, str]]: The conversation history
        """
        return self.messages

agent = Agent()

# Simple question-answer interaction
def chat_with_agent(question: str):
    print(f"\nUser: {question}")
    print(f"Assistant: {agent.process_query(question)}")

# Test MySQL connection
if __name__ == "__main__":
    # chat_with_agent("How many tasks do we have in our database?")
    # chat_with_agent("Which task priority is the highest?")
    chat_with_agent("List employees name and assigned task")
    # Admin/debug: run an INSERT statement
    # print(execute_sql("insert into TaskAssignee values(1, 10);"))