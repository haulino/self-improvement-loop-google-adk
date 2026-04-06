from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import LoopAgent, LlmAgent, SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext

# --- Constants ---
GEMINI_MODEL = "gemini-2.5-flash"

# --- State Keys ---
STATE_HIGH_LEVEL_DESC = "high_level_description"
STATE_CURRENT_PROMPT = "current_system_prompt"
STATE_PREVIOUS_PROMPT = "previous_system_prompt"
STATE_CURRENT_RESPONSE = "current_sample_response"
STATE_PREVIOUS_RESPONSE = "previous_sample_response"
STATE_CRITICISM = "criticism"
STATE_TEST_QUESTION = "test_question"

# Define the exact phrase the Critic should use to signal completion
COMPLETION_PHRASE = "Optimization complete: No further reductions possible without quality loss."

# --- Tool Definition ---
def exit_loop(tool_context: ToolContext):
    """Call this function ONLY when the critique indicates the prompt is perfectly optimized, signaling the iterative process should end."""
    print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    tool_context.actions.skip_summarization = True
    return {}

# --- Callbacks ---
def setup_initial_state(callback_context: CallbackContext):
    """Initialize the state with the goal and test question."""
    callback_context.state[STATE_HIGH_LEVEL_DESC] = "Expert technical support assistant that explains complex cloud architecture to beginners using simple analogies."
    callback_context.state[STATE_TEST_QUESTION] = "What is a Load Balancer?"
    callback_context.state[STATE_PREVIOUS_PROMPT] = "None (Initial iteration)"
    callback_context.state[STATE_PREVIOUS_RESPONSE] = "None (Initial iteration)"

# --- Agent Definitions ---

# STEP 1: Initial Prompt Generator (Runs ONCE)
initial_generator = LlmAgent(
    name="InitialPromptGenerator",
    model=GEMINI_MODEL,
    instruction=f"""
    You are a Prompt Engineer. Based on the description below, create a comprehensive and effective system prompt.
    Description: {{{STATE_HIGH_LEVEL_DESC}}}

    Output ONLY the system prompt text.
    """,
    output_key=STATE_CURRENT_PROMPT
)

# STEP 2: Response Tester (Inside Loop) - Tests the current prompt
response_tester = LlmAgent(
    name="ResponseTester",
    model=GEMINI_MODEL,
    instruction=f"""
    You are an LLM acting under the following system prompt:
    ---
    {{{STATE_CURRENT_PROMPT}}}
    ---

    Question to answer: {{{STATE_TEST_QUESTION}}}

    Output ONLY your response.
    """,
    output_key=STATE_CURRENT_RESPONSE
)

# STEP 3: Linguist Critic (Inside Loop)
linguist_critic = LlmAgent(
    name="LinguistCritic",
    model=GEMINI_MODEL,
    instruction=f"""
    You are a Linguist Critic. Your goal is to ensure system prompts and their outputs are concise without losing quality.

    **Current System Prompt:**
    ```
    {{{STATE_CURRENT_PROMPT}}}
    ```
    **Current Sample Response:**
    ```
    {{{STATE_CURRENT_RESPONSE}}}
    ```

    **Previous System Prompt:**
    ```
    {{{STATE_PREVIOUS_PROMPT}}}
    ```
    **Previous Sample Response:**
    ```
    {{{STATE_PREVIOUS_RESPONSE}}}
    ```

    **Your Tasks:**
    1. Compare the 'Current' versions with the 'Previous' versions.
    2. Check if the current prompt and response are significantly more concise than the previous ones.
    3. Ensure the quality, tone, and accuracy of the response are maintained or improved.
    4. Identify linguistic "fluff," redundancies, or over-explaining in the system prompt.

    IF the prompt can be further shortened while keeping response quality high, provide specific linguistic feedback.
    IF no further meaningful reduction can be made without sacrificing quality (i.e., it is perfectly optimized), respond EXACTLY with: "{COMPLETION_PHRASE}"

    Output ONLY your critique or the completion phrase.
    """,
    output_key=STATE_CRITICISM
)

# STEP 4: Prompt Optimizer (Inside Loop)
prompt_optimizer = LlmAgent(
    name="PromptOptimizer",
    model=GEMINI_MODEL,
    instruction=f"""
    You are a Prompt Optimizer. Your task is to rewrite the system prompt to be shorter and more efficient based on the Critic's feedback.

    **Current System Prompt:**
    ```
    {{{STATE_CURRENT_PROMPT}}}
    ```
    **Linguist Feedback:**
    {{{STATE_CRITICISM}}}

    **Your Task:**
    1. If the feedback is EXACTLY "{COMPLETION_PHRASE}", call the 'exit_loop' function immediately.
    2. Otherwise, rewrite the system prompt to be significantly more concise while incorporating the feedback.

    Output ONLY the optimized system prompt text OR call 'exit_loop'.
    """,
    tools=[exit_loop],
    output_key=STATE_CURRENT_PROMPT
)

def update_previous_state(callback_context: CallbackContext):
    """Store current values as 'previous' for the next iteration comparison."""
    callback_context.state[STATE_PREVIOUS_PROMPT] = callback_context.state.get(STATE_CURRENT_PROMPT, "")
    callback_context.state[STATE_PREVIOUS_RESPONSE] = callback_context.state.get(STATE_CURRENT_RESPONSE, "")

# STEP 5: The Refinement Loop
refinement_loop = LoopAgent(
    name="PromptOptimizationLoop",
    sub_agents=[
        response_tester,
        linguist_critic,
        prompt_optimizer
    ],
    after_agent_callback=update_previous_state,
    max_iterations=5
)

# STEP 6: Root Pipeline
root_agent = SequentialAgent(
    name="SelfImprovementPipeline",
    sub_agents=[
        initial_generator,
        refinement_loop
    ],
    before_agent_callback=setup_initial_state,
    description="Iteratively optimizes a system prompt for token efficiency and quality."
)

if __name__ == "__main__":
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    import asyncio

    async def main():
        session_service = InMemorySessionService()
        runner = Runner(
            app_name="self-improvement",
            agent=root_agent,
            session_service=session_service,
        )

        user_id = "user"
        session_id = "session"
        await session_service.create_session(
            app_name="self-improvement",
            user_id=user_id,
            session_id=session_id,
        )

        from google.genai import types
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text="Start optimization")]),
        ):
            if hasattr(event, 'content') and event.content:
                print(f"[{event.author}] {event.content}")

        session = await session_service.get_session(
            app_name="self-improvement",
            user_id=user_id,
            session_id=session_id,
        )
        print("\n--- Final Optimized Prompt ---")
        print(session.state.get(STATE_CURRENT_PROMPT))
        print("\n--- Final Sample Response ---")
        print(session.state.get(STATE_CURRENT_RESPONSE))

    asyncio.run(main())
