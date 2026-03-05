"""All LLM prompt templates for the newsletter system."""

SYSTEM_PROMPT = """You are an AI news curator. You analyze articles about AI and machine learning,
producing structured summaries, categorizations, and importance rankings."""

# Topics that articles can be tagged with
ALL_TOPICS = [
    "NLP", "Computer Vision", "Reinforcement Learning", "ML Theory",
    "AI Safety", "Multimodal", "Robotics", "AI Agents",
    "LLM APIs", "AI Infrastructure", "AI Startups", "Enterprise AI",
    "AI Regulation", "Fundraising",
    "Open Source Models", "AI Art", "AI Coding Tools", "AI Hardware", "Tutorials",
    "General AI",
]

CATEGORIES = ["research", "product", "industry", "open_source", "opinion"]
DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced"]

BATCH_PROCESS_PROMPT = """Analyze the following {count} articles about AI/ML. For each article, provide:

1. **category**: One of: {categories}
2. **topics**: 1-3 topics from this EXACT list (you MUST pick at least 1, use "General AI" if nothing else fits): {topics}
3. **difficulty_level**: One of: {difficulty_levels}
4. **base_importance_score**: Float from 1.0 to 10.0 (10 = groundbreaking, 1 = low relevance)
5. **summary_student**: A technical summary (2-3 sentences) aimed at ML students. Include methodology details, reference techniques.
6. **summary_industry**: A business-focused summary (2-3 sentences) aimed at industry professionals. Highlight practical impact, market implications.
7. **summary_enthusiast**: An accessible summary (2-3 sentences) aimed at AI enthusiasts. Explain concepts simply, highlight what's exciting.
8. **key_entities**: List of key entities mentioned (people, companies, models, datasets).

Respond with a JSON array. Each element must have these exact keys:
- "index" (int, 0-based matching the article order below)
- "category" (string)
- "topics" (array of strings)
- "difficulty_level" (string)
- "base_importance_score" (float)
- "summary_student" (string)
- "summary_industry" (string)
- "summary_enthusiast" (string)
- "key_entities" (array of strings)

ARTICLES:
{articles}

Respond ONLY with the JSON array, no other text."""

LEVEL_ADAPT_BEGINNER_PROMPT = """Rewrite this summary for someone new to AI/ML.
- Explain any technical terms inline
- Add a brief "Why this matters" context
- Make it 3-4 sentences
- Use simple, clear language

Original summary:
{summary}

Rewritten summary:"""

LEVEL_ADAPT_ADVANCED_PROMPT = """Condense this summary for an AI/ML expert.
- Use precise technical terminology without explanation
- Remove background context — assume the reader knows the field
- Keep it to 1-2 concise sentences
- Focus only on the key technical contribution or news

Original summary:
{summary}

Condensed summary:"""
