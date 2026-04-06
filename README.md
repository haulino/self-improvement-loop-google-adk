## Self-Improvement Loop

### Prompt Self-Improvement Loop
A tool designed to iteratively optimize LLM system prompts for token efficiency and response quality. It uses a "Linguist Critic" to ensure that as the prompt gets shorter, the quality remains high.

#### How to run:

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Google API credentials by adding your key to `.env`:
   ```
   GOOGLE_API_KEY=your-api-key
   ```

4. Run the script:
   ```bash
   python self-improvement.py
   ```

5. The script will:
   - Generate an initial prompt from a high-level description.
   - Iteratively test the prompt against a sample question.
   - Critique and rewrite the prompt to reduce its length and the length of its responses.
   - Print the final optimized prompt and sample response once the loop completes.