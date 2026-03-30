"""
=============================================================
STEP 4: LLM Generation with GPT API
=============================================================
This module handles:
- Taking retrieved context + user query
- Constructing a sales/marketing-optimized prompt
- Calling OpenAI GPT API for response generation
- Formatting the final response

Run this file standalone to test generation:
    python step4_generate.py
=============================================================
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ── Configuration ──────────────────────────────────────────
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")  # Use gpt-4o for best quality
TEMPERATURE = 0.3        # Low temp for factual, consistent responses
MAX_TOKENS = 1024        # Max response length


# ── 4A: System Prompts for Different Use Cases ────────────

SALES_SYSTEM_PROMPT = """You are an expert sales assistant for Pigment Company Orgochem Limited, 
a leading Indian company specializing in colorants, pigments, dyes, and pigment dispersions 
with 50+ years of experience.

Your role is to:
1. Answer customer queries about Pigment Company's products accurately using ONLY the provided context
2. Highlight relevant product advantages, certifications, and technical specifications
3. Recommend specific products (with grades and C.I. numbers) when applicable
4. Be professional yet approachable — like a knowledgeable sales engineer
5. If the context doesn't contain enough information to answer, say so honestly rather than guessing

Key company strengths to weave in naturally:
- 50+ years of expertise in colorants
- ISO 9001:2015 certified, REACH compliant, FDA approved
- Global presence across 40+ countries
- Strong R&D with nano-level particle control (down to 200nm)
- Comprehensive product range: Pigmeron (powders), Pigmefine (decorative), 
  Pigmetint (POS tinting), Pigmeperse (universal stainers), Pigmeflex (flexo inks)

Format guidelines:
- Use clear paragraphs, not excessive bullet points
- Include specific product names and grades when relevant
- End with a subtle call to action when appropriate
- Keep technical details accurate — these are industry professionals asking
"""

MARKETING_SYSTEM_PROMPT = """You are a marketing content specialist for Pigment Company Orgochem Limited,
a pioneer in the colorants industry since 1971.

Using ONLY the provided context, create compelling marketing content that:
1. Emphasizes Pigment Company's unique value propositions
2. Uses industry-specific terminology correctly
3. Highlights innovation (nano-level particle control, modern R&D)
4. Showcases the breadth of applications (ink, paint, plastic, textile)
5. Maintains a professional yet engaging tone

Always ground your content in the factual details from the context.
"""

TECHNICAL_SYSTEM_PROMPT = """You are a technical support specialist for Pigment Company Orgochem Limited.

Using ONLY the provided context, provide accurate technical information about:
1. Product specifications (C.I. numbers, grades, heat stability, particle size)
2. Application recommendations
3. Selection criteria for specific use cases
4. Fastness properties, chemical resistance, and compatibility

Be precise with numbers and specifications. If you're unsure about a detail, 
state that clearly rather than approximating.
"""


# ── 4B: Prompt Construction ───────────────────────────────
def build_prompt(
    query: str,
    context: str,
    conversation_history: Optional[List[Dict]] = None,
    mode: str = "sales"  # "sales", "marketing", "technical"
) -> List[Dict]:
    """
    Construct the full message array for the GPT API.
    
    Args:
        query: User's question
        context: Retrieved document context from Step 3
        conversation_history: Previous messages for multi-turn
        mode: Which system prompt to use
    
    Returns:
        List of message dicts for OpenAI API
    """
    # Select system prompt
    system_prompts = {
        "sales": SALES_SYSTEM_PROMPT,
        "marketing": MARKETING_SYSTEM_PROMPT,
        "technical": TECHNICAL_SYSTEM_PROMPT,
    }
    system_prompt = system_prompts.get(mode, SALES_SYSTEM_PROMPT)
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history if multi-turn
    if conversation_history:
        messages.extend(conversation_history)
    
    # Construct the user message with context
    user_message = f"""Based on the following product documentation from Pigment Company Orgochem:

{context}

---
Customer Question: {query}

Please provide a helpful, accurate response based on the above documentation."""
    
    messages.append({"role": "user", "content": user_message})
    
    return messages


# ── 4C: GPT API Call ──────────────────────────────────────
class Pigment CompanyGenerator:
    """
    LLM response generator using OpenAI GPT API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = GPT_MODEL):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in .env file "
                "or pass it directly."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.conversation_history = []
        
        print(f"✅ Generator initialized with model: {model}")
    
    def generate(
        self,
        query: str,
        context: str,
        mode: str = "sales",
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        stream: bool = False
    ) -> str:
        """
        Generate a response using GPT API.
        
        Args:
            query: User's question
            context: Retrieved context from vector store
            mode: Response style (sales/marketing/technical)
            temperature: Creativity level (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum response length
            stream: Whether to stream the response
        
        Returns:
            Generated response text
        """
        messages = build_prompt(
            query=query,
            context=context,
            conversation_history=self.conversation_history,
            mode=mode
        )
        
        try:
            if stream:
                return self._stream_response(messages, temperature, max_tokens)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            answer = response.choices[0].message.content
            
            # Track usage
            usage = response.usage
            print(f"   📊 Tokens: {usage.prompt_tokens} prompt + "
                  f"{usage.completion_tokens} completion = {usage.total_tokens} total")
            
            # Update conversation history for multi-turn
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return answer
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def _stream_response(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Stream response token by token."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                print(token, end="", flush=True)
                full_response += token
        
        print()  # Newline after streaming
        return full_response
    
    def reset_conversation(self):
        """Clear conversation history for a fresh start."""
        self.conversation_history = []
        print("🔄 Conversation history cleared")


# ── 4D: Response Post-Processing ──────────────────────────
def format_response_with_sources(
    answer: str,
    results: List[Dict]
) -> str:
    """
    Add source references to the generated answer.
    Useful for transparency and verification.
    """
    sources = []
    for r in results:
        page = r["metadata"].get("page_number", "?")
        section = r["metadata"].get("section_title", "Unknown")
        sources.append(f"Page {page}: {section}")
    
    source_text = "\n".join(f"  [{i+1}] {s}" for i, s in enumerate(sources))
    
    return f"{answer}\n\n📚 Sources:\n{source_text}"


# ── Main ──────────────────────────────────────────────────
def test_generation():
    """Test the generation pipeline with mock context."""
    print("\n" + "="*60)
    print("STEP 4: LLM Generation Testing")
    print("="*60 + "\n")
    
    # Mock context (in real use, this comes from Step 3)
    mock_context = """
    --- Source 1 (Page 10 | Pigmeron Pigments for Plastic Moldings) ---
    C.I.NO: Blue 15.1, Product: Alpha Blue AFP, Grade: M-851, Heat Stability: 280°C
    C.I.NO: Blue 15.3, Product: Beta Blue BFP, Grade: M-702, Heat Stability: 280°C
    C.I.NO: Green 7, Product: Green GFP, Grade: M-900, Heat Stability: 300°C
    C.I.NO: Violet 23, Product: Violet BL, Grade: M-505, Heat Stability: 280°C
    
    --- Source 2 (Page 3 | About Pigment Company) ---
    A 50 year old renowned company in the field of Dyestuff, Pigments, Pigments Dispersions
    and Digital Inks. Knowledge and experience in Purification of Colorants and stable 
    Particle reduction up to 200 Nano meter.
    """
    
    query = "Which pigments do you recommend for high-temperature plastic molding above 250°C?"
    
    try:
        generator = Pigment CompanyGenerator()
        
        print(f"🔍 Query: {query}\n")
        answer = generator.generate(query, mock_context, mode="sales")
        print(f"\n💬 Response:\n{answer}")
        
    except ValueError as e:
        print(f"\n⚠️  {e}")
        print("   To test generation, create a .env file with your OPENAI_API_KEY")
        print("   For now, here's what the prompt would look like:\n")
        
        messages = build_prompt(query, mock_context, mode="sales")
        for msg in messages:
            print(f"   [{msg['role'].upper()}]")
            print(f"   {msg['content'][:300]}...\n")


if __name__ == "__main__":
    test_generation()
