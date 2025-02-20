from flask import Flask, request, jsonify
import asyncio
import os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory, ChatHistoryTruncationReducer
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import KernelFunctionSelectionStrategy
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import KernelFunctionTerminationStrategy
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments
from functools import wraps
from dotenv import load_dotenv
import os

load_dotenv()

# Update Azure OpenAI Configuration to use environment variables
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "default_key")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "default_endpoint")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "default_deployment")



app = Flask(__name__)


def create_kernel_with_chat_completion(service_id: str) -> Kernel:
    """Create a kernel with Azure OpenAI chat completion service"""
    kernel = Kernel()
    chat_service = AzureChatCompletion(
        service_id=service_id,
        deployment_name=AZURE_DEPLOYMENT_NAME,
        endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION
    )
    kernel.add_service(chat_service)
    return kernel

class MultiAgentOrchestrator:
    def __init__(self):
        # Initialize Program Manager
        self.program_manager = ChatCompletionAgent(
            service_id="ProgramManager",
            kernel=create_kernel_with_chat_completion("ProgramManager"),
            name="ProgramManager",
            instructions="""You are a Program Manager responsible for:
            1. ONLY analyzing the initial user requirements for the app
            2. Creating ONE clear, structured plan including features, timeline, and costs
            3. Hand over to the Software Engineer after ONE complete plan
            4. Do not repeat yourself or ask clarifying questions unless explicitly asked by the user

            Remember: Make ONE comprehensive response and then let the Software Engineer take over.
            """
        )

        # Initialize Software Engineer
        self.software_engineer = ChatCompletionAgent(
            service_id="SoftwareEngineer",
            kernel=create_kernel_with_chat_completion("SoftwareEngineer"),
            name="SoftwareEngineer",
            instructions="""You are a Software Engineer responsible for:
            1. Taking the Program Manager's plan and implementing the technical solution
            2. Creating the actual HTML and JavaScript code for the application
            3. Providing COMPLETE working code that can be reviewed
            4. Once code is complete, hand over to Project Manager for review

            Focus on delivering working code based on the requirements. Don't discuss - just implement.
            """
        )

        # Initialize Project Manager
        self.project_manager = ChatCompletionAgent(
            service_id="ProjectManager",
            kernel=create_kernel_with_chat_completion("ProjectManager"),
            name="ProjectManager",
            instructions="""You are a Project Manager responsible for:
            1. Reviewing the Software Engineer's implementation
            2. Verifying all requirements are met
            3. Either approve by responding ONLY with "approve" if everything is correct
            4. Or provide specific feedback to the Software Engineer if changes are needed

            Be thorough in review but concise in feedback. No discussion needed.
            """
        )

        # Create selection function for strict turn-taking
        self.selection_function = KernelFunctionFromPrompt(
            function_name="selection",
            prompt="""
            Select the next agent based on these strict rules:
            1. Program Manager ALWAYS goes first and ONLY once
            2. Software Engineer ALWAYS goes second with implementation
            3. Project Manager ALWAYS goes third to review
            4. If Project Manager says "approve", stop
            5. If Project Manager gives feedback, Software Engineer goes next
            
            Current conversation history: {{$history}}
            

            Respond with exactly one name: ProgramManager, SoftwareEngineer, or ProjectManager
            """
        )

        # Set up history reducer for efficiency
        self.history_reducer = ChatHistoryTruncationReducer(target_count=3)

        # Create group chat with selection and termination strategies
        self.group_chat = AgentGroupChat(
            agents=[self.program_manager, self.software_engineer, self.project_manager],
            selection_strategy=KernelFunctionSelectionStrategy(
                function=self.selection_function,
                kernel=create_kernel_with_chat_completion("selection"),
                result_parser=lambda result: str(result.value[0]) if result.value is not None else "ProgramManager",
                agent_variable_name="agents",
                history_variable_name="history",
                history_reducer=self.history_reducer
            ),
            termination_strategy=KernelFunctionTerminationStrategy(
                agents=[self.project_manager],
                function=KernelFunctionFromPrompt(
                    function_name="termination",
                    prompt="Return 'yes' ONLY if the message contains exactly 'approve'\nMessage: {{$history}}"
                ),
                kernel=create_kernel_with_chat_completion("termination"),
                result_parser=lambda result: "yes" in str(result.value[0]).lower() if result.value else False,
                history_variable_name="history",
                maximum_iterations=5,
                history_reducer=self.history_reducer
            )
        )

    async def process_request(self, user_input: str) -> list:
        """Process a user request through the agent group chat"""
        # Reset chat completion state
        self.group_chat.is_complete = False
        
        # Add user message to group chat
        await self.group_chat.add_chat_message(
            ChatMessageContent(role=AuthorRole.USER, content=user_input)
        )

        # Collect all responses
        responses = []
        async for response in self.group_chat.invoke():
            responses.append({
                'role': response.role,
                'content': response.content,
                'author': response.name or response.role
            })

        return responses

# Create orchestrator instance
orchestrator = MultiAgentOrchestrator()

def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

@app.route('/chat', methods=['POST'])
@async_route
async def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    try:
        responses = await orchestrator.process_request(data['message'])
        return jsonify({
            'status': 'success',
            'conversation': responses
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)















# >>>>>>>>  working but need verification of code.




# from flask import Flask, request, jsonify
# import asyncio
# import os
# from semantic_kernel import Kernel
# from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
# from semantic_kernel.contents import ChatHistory, ChatHistoryTruncationReducer
# from semantic_kernel.contents.chat_message_content import ChatMessageContent
# from semantic_kernel.contents.utils.author_role import AuthorRole
# from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
# from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
# from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import KernelFunctionSelectionStrategy
# from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import KernelFunctionTerminationStrategy
# from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
# from semantic_kernel.functions.kernel_arguments import KernelArguments
# from functools import wraps

# app = Flask(__name__)

# # Azure OpenAI Configuration
# AZURE_API_KEY = "ee65aca022a74803b2e2d1ff4c373b05"
# AZURE_ENDPOINT = "https://firstsource.openai.azure.com"
# AZURE_API_VERSION = "2024-02-15-preview"
# AZURE_DEPLOYMENT_NAME = "gpt-4o-v05-13"

# def create_kernel_with_chat_completion(service_id: str) -> Kernel:
#     """Create a kernel with Azure OpenAI chat completion service"""
#     kernel = Kernel()
#     chat_service = AzureChatCompletion(
#         service_id=service_id,
#         deployment_name=AZURE_DEPLOYMENT_NAME,
#         endpoint=AZURE_ENDPOINT,
#         api_key=AZURE_API_KEY,
#         api_version=AZURE_API_VERSION
#     )
#     kernel.add_service(chat_service)
#     return kernel

# class MultiAgentOrchestrator:
#     def __init__(self):
#         # Initialize Program Manager
#         self.program_manager = ChatCompletionAgent(
#             service_id="ProgramManager",
#             kernel=create_kernel_with_chat_completion("ProgramManager"),
#             name="ProgramManager",
#             instructions="""You are a Program Manager responsible for:
#             1. ONLY analyzing the initial user requirements for the app
#             2. Creating ONE clear, structured plan including features, timeline, and costs
#             3. Hand over to the Software Engineer after ONE complete plan
#             4. Do not repeat yourself or ask clarifying questions unless explicitly asked by the user

#             Remember: Make ONE comprehensive response and then let the Software Engineer take over.
#             """
#         )

#         # Initialize Software Engineer
#         self.software_engineer = ChatCompletionAgent(
#             service_id="SoftwareEngineer",
#             kernel=create_kernel_with_chat_completion("SoftwareEngineer"),
#             name="SoftwareEngineer",
#             instructions="""You are a Software Engineer responsible for:
#             1. Taking the Program Manager's plan and implementing the technical solution
#             2. Creating the actual HTML and JavaScript code for the application
#             3. Providing COMPLETE working code that can be reviewed
#             4. Once code is complete, hand over to Project Manager for review

#             Focus on delivering working code based on the requirements. Don't discuss - just implement.
#             """
#         )

#         # Initialize Project Manager
#         self.project_manager = ChatCompletionAgent(
#             service_id="ProjectManager",
#             kernel=create_kernel_with_chat_completion("ProjectManager"),
#             name="ProjectManager",
#             instructions="""You are a Project Manager responsible for:
#             1. Reviewing the Software Engineer's implementation
#             2. Verifying all requirements are met
#             3. Either approve by responding ONLY with "approve" if everything is correct
#             4. Or provide specific feedback to the Software Engineer if changes are needed

#             Be thorough in review but concise in feedback. No discussion needed.
#             """
#         )

#         # Create selection function for strict turn-taking
#         self.selection_function = KernelFunctionFromPrompt(
#             function_name="selection",
#             prompt="""
#             Select the next agent based on these strict rules:
#             1. Program Manager ALWAYS goes first and ONLY once
#             2. Software Engineer ALWAYS goes second with implementation
#             3. Project Manager ALWAYS goes third to review
#             4. If Project Manager says "approve", stop
#             5. If Project Manager gives feedback, Software Engineer goes next
            
#             Current conversation history: {{$history}}
#             Last message was from: {{$last_author}}

#             Respond with exactly one name: ProgramManager, SoftwareEngineer, or ProjectManager
#             """
#         )

#         # Set up history reducer for efficiency
#         self.history_reducer = ChatHistoryTruncationReducer(target_count=3)

#         # Create group chat with selection and termination strategies
#         self.group_chat = AgentGroupChat(
#             agents=[self.program_manager, self.software_engineer, self.project_manager],
#             selection_strategy=KernelFunctionSelectionStrategy(
#                 function=self.selection_function,
#                 kernel=create_kernel_with_chat_completion("selection"),
#                 result_parser=lambda result: str(result.value[0]) if result.value is not None else "ProgramManager",
#                 agent_variable_name="agents",
#                 history_variable_name="history",
#                 history_reducer=self.history_reducer
#             ),
#             termination_strategy=KernelFunctionTerminationStrategy(
#                 agents=[self.project_manager],
#                 function=KernelFunctionFromPrompt(
#                     function_name="termination",
#                     prompt="Return 'yes' ONLY if the message contains exactly 'approve'\nMessage: {{$history}}"
#                 ),
#                 kernel=create_kernel_with_chat_completion("termination"),
#                 result_parser=lambda result: "yes" in str(result.value[0]).lower() if result.value else False,
#                 history_variable_name="history",
#                 maximum_iterations=5,
#                 history_reducer=self.history_reducer
#             )
#         )

#     async def process_request(self, user_input: str) -> list:
#         """Process a user request through the agent group chat"""
#         # Reset chat completion state
#         self.group_chat.is_complete = False
        
#         # Add user message to group chat
#         await self.group_chat.add_chat_message(
#             ChatMessageContent(role=AuthorRole.USER, content=user_input)
#         )

#         # Collect all responses
#         responses = []
#         async for response in self.group_chat.invoke():
#             responses.append({
#                 'role': response.role,
#                 'content': response.content,
#                 'author': response.name or response.role
#             })

#         return responses

# # Create orchestrator instance
# orchestrator = MultiAgentOrchestrator()

# def async_route(f):
#     @wraps(f)
#     def wrapped(*args, **kwargs):
#         return asyncio.run(f(*args, **kwargs))
#     return wrapped

# @app.route('/chat', methods=['POST'])
# @async_route
# async def chat():
#     data = request.get_json()
#     if not data or 'message' not in data:
#         return jsonify({'error': 'No message provided'}), 400

#     try:
#         responses = await orchestrator.process_request(data['message'])
#         return jsonify({
#             'status': 'success',
#             'conversation': responses
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({'status': 'healthy'})

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)




















# >>> working but need to update kernal and seasion id 



# from flask import Flask, request, jsonify
# import os
# from semantic_kernel import Kernel
# from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatPromptExecutionSettings
# from semantic_kernel.contents import ChatHistory
# from dataclasses import dataclass
# from typing import List, Optional
# import asyncio
# from functools import wraps

# app = Flask(__name__)

# # Configuration
# AZURE_API_KEY = "ee65aca022a74803b2e2d1ff4c373b05"
# AZURE_ENDPOINT = "https://firstsource.openai.azure.com"
# AZURE_API_VERSION = "2024-02-15-preview"
# AZURE_DEPLOYMENT_NAME = "gpt-4o-v05-13"

# @dataclass
# class ChatMessage:
#     role: str
#     content: str
#     author_name: Optional[str] = None

# class Agent:
#     def __init__(self, name: str, instructions: str, chat_completion_service: AzureChatCompletion):
#         self.name = name
#         self.instructions = instructions
#         self.chat_completion_service = chat_completion_service

#     async def process_message(self, message: str) -> str:
#         chat_history = ChatHistory()
#         chat_history.add_system_message(self.instructions)
#         chat_history.add_user_message(message)
        
#         execution_settings = OpenAIChatPromptExecutionSettings()
        
#         response = await self.chat_completion_service.get_chat_message_content(
#             chat_history=chat_history,
#             settings=execution_settings,
#         )
        
#         return response.content

# class ApprovalTerminationStrategy:
#     def __init__(self, project_manager_agent: Agent, max_iterations: int = 6):
#         self.project_manager = project_manager_agent
#         self.max_iterations = max_iterations
#         self.current_iteration = 0

#     def should_terminate(self, last_message: ChatMessage) -> bool:
#         self.current_iteration += 1
#         return (
#             self.current_iteration >= self.max_iterations or
#             (last_message.author_name == self.project_manager.name and
#              "approve" in last_message.content.lower())
#         )

# class AgentGroupChat:
#     def __init__(self, program_manager: Agent, software_engineer: Agent, project_manager: Agent):
#         self.agents = [program_manager, software_engineer, project_manager]
#         self.chat_history: List[ChatMessage] = []
#         self.termination_strategy = ApprovalTerminationStrategy(project_manager)

#     def add_message(self, message: ChatMessage):
#         self.chat_history.append(message)

#     async def run_conversation(self, initial_prompt: str) -> List[ChatMessage]:
#         # Add initial user prompt
#         self.add_message(ChatMessage(role="user", content=initial_prompt))

#         current_agent_idx = 0
#         while True:
#             current_agent = self.agents[current_agent_idx]
            
#             # Process the last message through current agent
#             response = await current_agent.process_message(self.chat_history[-1].content)
            
#             # Add agent's response to chat history
#             message = ChatMessage(
#                 role="assistant",
#                 content=response,
#                 author_name=current_agent.name
#             )
#             self.add_message(message)

#             # Check if we should terminate
#             if self.termination_strategy.should_terminate(message):
#                 break

#             # Move to next agent
#             current_agent_idx = (current_agent_idx + 1) % len(self.agents)

#         return self.chat_history

# # Create a single instance of the chat completion service
# chat_completion_service = AzureChatCompletion(
#     deployment_name=AZURE_DEPLOYMENT_NAME,
#     endpoint=AZURE_ENDPOINT,
#     api_key=AZURE_API_KEY,
#     api_version=AZURE_API_VERSION
# )

# # Define agent instructions
# PROGRAM_MANAGER_INSTRUCTIONS = """
# You are a program manager which will take the requirement and create a plan for creating app. 
# Program Manager understands the user requirements and form the detail documents with requirements and costing.
# """

# SOFTWARE_ENGINEER_INSTRUCTIONS = """
# You are Software Engineer, and your goal is to create web app using HTML and JavaScript by taking into consideration all
# the requirements given by Program Manager.
# """

# PROJECT_MANAGER_INSTRUCTIONS = """
# You are manager which will review software engineer code, and make sure all client requirements are completed.
# You are the guardian of quality, ensuring the final product meets all specifications and receives the green light for release.
# Once all client requirements are completed, you can approve the request by just responding "approve"
# """

# # Create agents with the chat completion service
# program_manager = Agent("ProgramManager", PROGRAM_MANAGER_INSTRUCTIONS, chat_completion_service)
# software_engineer = Agent("SoftwareEngineer", SOFTWARE_ENGINEER_INSTRUCTIONS, chat_completion_service)
# project_manager = Agent("ProjectManager", PROJECT_MANAGER_INSTRUCTIONS, chat_completion_service)

# # Create group chat
# group_chat = AgentGroupChat(program_manager, software_engineer, project_manager)

# def async_route(f):
#     @wraps(f)
#     def wrapped(*args, **kwargs):
#         return asyncio.run(f(*args, **kwargs))
#     return wrapped

# @app.route('/chat', methods=['POST'])
# @async_route
# async def chat():
#     data = request.get_json()
#     if not data or 'message' not in data:
#         return jsonify({'error': 'No message provided'}), 400

#     try:
#         # Run the conversation
#         chat_history = await group_chat.run_conversation(data['message'])

#         # Format the response
#         formatted_history = [
#             {
#                 'role': msg.role,
#                 'content': msg.content,
#                 'author': msg.author_name if msg.author_name else msg.role
#             }
#             for msg in chat_history
#         ]

#         return jsonify({
#             'status': 'success',
#             'conversation': formatted_history
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({'status': 'healthy'})

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=8550)














# # app.py
# from flask import Flask, request, jsonify
# import semantic_kernel as sk
# from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
# from semantic_kernel.contents.chat_message_content import ChatMessageContent
# from semantic_kernel.contents.utils.author_role import AuthorRole
# from semantic_kernel.agents import ChatCompletionAgent
# from semantic_kernel.functions.kernel_function_decorator import kernel_function
# from dotenv import load_dotenv
# import os
# import asyncio
# from typing import List, Dict, Any

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)

# # Define agent names
# SUPER_AGENT_NAME = "SuperAgent"
# OPERATIONAL_AGENT_NAME = "OperationalAgent"
# WRITER_AGENT_NAME = "WriterAgent"

# def create_kernel_with_chat_completion(service_id: str) -> sk.Kernel:
#     kernel = sk.Kernel()
#     kernel.add_service(
#         AzureChatCompletion(
#             service_id=service_id,
#             deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
#             endpoint=os.getenv("AZURE_ENDPOINT"),
#             api_key=os.getenv("AZURE_API_KEY"),
#             api_version=os.getenv("AZURE_API_VERSION")
#         )
#     )
#     return kernel

# class EmailAgentSystem:
#     def __init__(self):
#         # Initialize agents
#         self.super_agent = ChatCompletionAgent(
#             service_id=SUPER_AGENT_NAME,
#             kernel=create_kernel_with_chat_completion(SUPER_AGENT_NAME),
#             name=SUPER_AGENT_NAME,
#             instructions="""
#             You are the first agent in the chain. Your role is to analyze email inquiries.
            
#             Instructions:
#             1. Carefully read and understand the email content
#             2. Identify main issues and key requirements
#             3. Provide a clear, structured summary of:
#                - Primary issue
#                - User requirements
#                - Urgency level
#                - Any additional context
            
#             Format your response as:
#             "ANALYSIS:
#             [Your detailed analysis here]
            
#             END_ANALYSIS"
#             """
#         )

#         self.operational_agent = ChatCompletionAgent(
#             service_id=OPERATIONAL_AGENT_NAME,
#             kernel=create_kernel_with_chat_completion(OPERATIONAL_AGENT_NAME),
#             name=OPERATIONAL_AGENT_NAME,
#             instructions="""
#             You are the second agent in the chain. Your role is to provide solutions based on the analysis.
            
#             Instructions:
#             1. Review the analysis from the Super Agent
#             2. Develop practical, actionable solutions
#             3. Provide step-by-step recommendations
#             4. Include specific actions required
            
#             Format your response as:
#             "SOLUTION:
#             [Your detailed solution here]
            
#             END_SOLUTION"
#             """
#         )

#         self.writer_agent = ChatCompletionAgent(
#             service_id=WRITER_AGENT_NAME,
#             kernel=create_kernel_with_chat_completion(WRITER_AGENT_NAME),
#             name=WRITER_AGENT_NAME,
#             instructions="""
#             You are the final agent in the chain. Your role is to write a professional email response.
            
#             Instructions:
#             1. Use the analysis and solutions provided
#             2. Write a clear, professional email that:
#                - Acknowledges the issue
#                - Presents the solution
#                - Provides next steps
#                - Maintains a professional tone
            
#             Format your response as:
#             "EMAIL_RESPONSE:
#             [Your email response here]
            
#             END_EMAIL"
#             """
#         )

#         self.agents = [self.super_agent, self.operational_agent, self.writer_agent]

#     async def get_agent_response(self, agent: ChatCompletionAgent, messages: List[Dict[str, Any]]) -> str:
#         try:
#             # Convert messages to chat format
#             chat_messages = []
            
#             # Add agent instructions as system message
#             chat_messages.append(
#                 ChatMessageContent(
#                     role=AuthorRole.SYSTEM,
#                     content=agent.instructions
#                 )
#             )
            
#             # Add conversation history
#             for msg in messages:
#                 if isinstance(msg, dict):
#                     chat_messages.append(
#                         ChatMessageContent(
#                             role=msg.get("role", AuthorRole.USER),
#                             content=msg.get("content", ""),
#                             name=msg.get("name", None)
#                         )
#                     )
#                 else:
#                     chat_messages.append(msg)

#             # Call the chat completion directly using the kernel
#             semantic_function = agent.kernel.create_semantic_function(
#                 prompt_template=agent.instructions,
#                 max_tokens=2000,
#                 temperature=0.7,
#                 top_p=1.0
#             )
#             context = agent.kernel.create_new_context()
#             for msg in chat_messages:
#                 context.variables[f"message_{len(context.variables)}"] = msg.content
            
#             result = await semantic_function.invoke_async(context=context)
#             return str(result)

#         except Exception as e:
#             raise Exception(f"Error getting response from {agent.name}: {str(e)}")

#     async def process_email(self, email_content: str) -> dict:
#         try:
#             messages = [{"role": AuthorRole.USER, "content": email_content}]
#             responses = []

#             # Step 1: Analysis from Super Agent
#             messages_for_super = messages.copy()
#             analysis = await self.get_agent_response(self.super_agent, messages_for_super)
#             responses.append({"agent": SUPER_AGENT_NAME, "content": analysis})
            
#             # Step 2: Solution from Operational Agent
#             messages_for_operational = messages.copy()
#             messages_for_operational.append({"role": AuthorRole.ASSISTANT, "content": analysis, "name": SUPER_AGENT_NAME})
#             solution = await self.get_agent_response(self.operational_agent, messages_for_operational)
#             responses.append({"agent": OPERATIONAL_AGENT_NAME, "content": solution})
            
#             # Step 3: Email composition from Writer Agent
#             messages_for_writer = messages.copy()
#             messages_for_writer.append({"role": AuthorRole.ASSISTANT, "content": analysis, "name": SUPER_AGENT_NAME})
#             messages_for_writer.append({"role": AuthorRole.ASSISTANT, "content": solution, "name": OPERATIONAL_AGENT_NAME})
#             email_response = await self.get_agent_response(self.writer_agent, messages_for_writer)
#             responses.append({"agent": WRITER_AGENT_NAME, "content": email_response})

#             return {
#                 "status": "success",
#                 "analysis": analysis,
#                 "solution": solution,
#                 "email_response": email_response,
#                 "full_conversation": responses
#             }

#         except Exception as e:
#             raise Exception(f"Error processing email: {str(e)}")

# # Initialize the agent system
# agent_system = EmailAgentSystem()

# @app.route('/process_email', methods=['POST'])
# async def process_email():
#     try:
#         data = request.get_json()
#         if not data or 'email_content' not in data:
#             return jsonify({"error": "Missing email_content in request"}), 400

#         result = await agent_system.process_email(data['email_content'])
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# def run_async_app():
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     app.run(debug=True, use_reloader=False)

# if __name__ == '__main__':
#     run_async_app()
