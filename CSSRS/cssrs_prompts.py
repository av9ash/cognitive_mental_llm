from datetime import time
import json

exp_temp = {'Label': 'Final Classification'}


def zero_shot_cot(input_text):
    """Zero-shot Chain-of-Thought (CoT) prompting for Reddit post classification."""
    prompt = f"""
    Analyze the following Reddit post and determine the mental health classification. Think step by step before providing the final classification.

    Post: "{input_text}"

    Step-by-step reasoning:
    1. Identify key phrases and emotions in the text.
    2. Assess whether the post expresses distress, support, or suicidal intent.
    3. Consider whether the post includes thoughts (ideation), behaviors, or past attempts.
    4. Based on this reasoning, classify the post into one of the following categories:

    - Supportive: If the post offers encouragement or support.
    - Ideation: If the post expresses suicidal thoughts or feelings.
    - Behavior: If the post describes suicidal behaviors but not an attempt.
    - Attempt: If the post mentions a past suicide attempt.
    - Indicator: If the post suggests signs of distress but does not fit the above.

    Provide a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt


def few_shot_cot(input_text):
    """Few-shot Chain-of-Thought (CoT) prompting for Reddit post classification."""

    example_prompts = """
    Example 1:
    Post: "I'm really struggling lately, but I know there are people who care about me. I just want to remind everyone that you're not alone."
    Step-by-step reasoning:
    1. The user acknowledges struggling but focuses on hope and encouragement.
    2. The post is supportive rather than expressing suicidal thoughts or behavior.
    3. Classification: Supportive

    Example 2:
    Post: "I've been feeling like there's no way out. I think about ending it all the time."
    Step-by-step reasoning:
    1. The post clearly expresses suicidal thoughts.
    2. No mention of actions or past attempts, only ideation.
    3. Classification: Ideation

    Example 3:
    Post: "I spent the night researching different ways to end my life, but I'm still here."
    Step-by-step reasoning:
    1. The post discusses researching suicide methods but no action taken yet.
    2. This suggests suicidal behavior without an actual attempt.
    3. Classification: Behavior

    Example 4:
    Post: "Last year, I attempted to overdose but survived. I'm not sure how I feel about it."
    Step-by-step reasoning:
    1. The post explicitly states a past suicide attempt.
    2. The person is reflecting on their experience.
    3. Classification: Attempt
    """

    prompt = f"""
    Below are examples of how to classify Reddit posts based on mental health signals. Follow the same reasoning to classify the new post.

    {example_prompts}

    Now analyze the following post:

    Post: "{input_text}"

    Step-by-step reasoning:
    1. Identify key phrases and emotions in the text.
    2. Assess whether the post expresses distress, support, or suicidal intent.
    3. Consider whether the post includes thoughts (ideation), behaviors, or past attempts.
    4. Based on this reasoning, classify the post into one of the categories: Supportive, Ideation, Behavior, Attempt, Indicator.

    Provide a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt


def role_based_prompt(input_text, role="therapist"):
    """Role-based prompting for classifying Reddit posts into mental health categories."""

    roles = {
        "therapist": """
        You are a licensed therapist analyzing a Reddit post to assess mental health risk.
        Consider psychological indicators such as distress, suicidal intent, or supportive language.
        """
    }

    prompt = f"""
    {roles.get(role, roles['therapist'])}

    Post: "{input_text}"

    Step-by-step reasoning:
    1. Analyze the language and emotions expressed in the post.
    2. Identify signs of distress, support, or suicidal thoughts/behavior.
    3. Categorize the post into one of the following:
       - Supportive: If it offers encouragement or support.
       - Ideation: If it expresses suicidal thoughts.
       - Behavior: If it describes suicidal behaviors without an attempt.
       - Attempt: If it mentions a past suicide attempt.
       - Indicator: If it suggests distress but does not fit the above.

    Provide a structured JSON response with the following format: {json.dumps(exp_temp)}.
    """
    return prompt


def tree_of_thoughts(input_text):
    """Tree-of-Thought (ToT) prompting for Reddit post classification."""

    prompt = f"""
    You are an AI language model trained to classify mental health-related Reddit posts using a **Tree of Thoughts (ToT) approach**.
    Instead of a single reasoning path, generate **three different reasoning trees** to explore different perspectives.

    Post: "{input_text}"

    **Tree 1: Emotional & Sentiment Analysis**
    - Identify key emotional tones (e.g., sadness, hopelessness, encouragement).
    - Does the post express distress or a coping mechanism?
    - Determine if emotional distress aligns with suicidal thoughts or support.

    **Tree 2: Behavioral Indicators**
    - Does the post discuss suicidal behaviors (e.g., research, plans, attempts)?
    - Does it mention external factors (e.g., therapy, social support, medications)?
    - Is there evidence of past suicidal attempts?

    **Tree 3: Risk & Protective Factors**
    - Does the post include protective factors (e.g., hope, future planning, social support)?
    - Are there clear indicators of increased suicide risk?
    - How severe are the linguistic indicators of distress?

    **Evaluation & Final Classification**
    - Compare insights from all three thought trees.
    - Determine the most supported classification from:
      - Supportive
      - Ideation
      - Behavior
      - Attempt
      - Indicator

    Provide a structured JSON response summarizing each thought tree and the final classification, following this format:

    {json.dumps({
        "Tree 1 - Emotional Analysis": "...",
        "Tree 2 - Behavioral Indicators": "...",
        "Tree 3 - Risk & Protective Factors": "...",
        "Label": "Final Classification"
    }, indent=4)}
    """

    return prompt

def hybrid_tot_few_shot(input_text):
    """Hybrid Tree-of-Thought (ToT) and Few-Shot Chain-of-Thought (CoT) prompting for Reddit post classification."""

    example_prompts = """
    Example 1:
    Post: "I'm really struggling lately, but I know there are people who care about me. I just want to remind everyone that you're not alone."
    Step-by-step reasoning:
    - Emotional Analysis: Expresses struggle but focuses on hope and encouragement.
    - Behavioral Indicators: No mention of suicidal ideation or behaviors.
    - Risk & Protective Factors: Presence of hope and social support reduces risk.
    - **Classification: Supportive**

    Example 2:
    Post: "I've been feeling like there's no way out. I think about ending it all the time."
    Step-by-step reasoning:
    - Emotional Analysis: Strong expression of despair and hopelessness.
    - Behavioral Indicators: No suicidal actions, but consistent ideation.
    - Risk & Protective Factors: No protective factors mentioned.
    - **Classification: Ideation**

    Example 3:
    Post: "I spent the night researching different ways to end my life, but I'm still here."
    Step-by-step reasoning:
    - Emotional Analysis: Despair and possible intent to act.
    - Behavioral Indicators: Mentions researching suicide methods.
    - Risk & Protective Factors: No mention of protective factors.
    - **Classification: Behavior**

    Example 4:
    Post: "Last year, I attempted to overdose but survived. I'm not sure how I feel about it."
    Step-by-step reasoning:
    - Emotional Analysis: Reflects on past suicidal behavior.
    - Behavioral Indicators: Explicit mention of a past suicide attempt.
    - Risk & Protective Factors: Uncertainty about the future.
    - **Classification: Attempt**
    """

    prompt = f"""
    You are an AI model trained to classify mental health-related Reddit posts using a **Hybrid Tree-of-Thought (ToT) and Few-Shot Learning approach**.

    **Methodology**: 
    - Instead of a single reasoning path, generate **three different reasoning trees** to analyze the post from multiple perspectives.
    - Use past examples to ensure logical and consistent classification.

    **Examples:**
    {example_prompts}

    Now analyze the following post:

    **Post:** "{input_text}"

    **Tree 1: Emotional & Sentiment Analysis**
    - Identify key emotional tones (e.g., sadness, hopelessness, encouragement).
    - Does the post express distress or a coping mechanism?
    - Determine if emotional distress aligns with suicidal thoughts or support.

    **Tree 2: Behavioral Indicators**
    - Does the post discuss suicidal behaviors (e.g., research, plans, attempts)?
    - Does it mention external factors (e.g., therapy, social support, medications)?
    - Is there evidence of past suicidal attempts?

    **Tree 3: Risk & Protective Factors**
    - Does the post include protective factors (e.g., hope, future planning, social support)?
    - Are there clear indicators of increased suicide risk?
    - How severe are the linguistic indicators of distress?

    **Final Evaluation & Classification**
    - Compare insights from all three trees.
    - Based on the reasoning patterns in previous examples, determine the most supported classification:
      - **Supportive**
      - **Ideation**
      - **Behavior**
      - **Attempt**
      - **Indicator**

    Provide a structured JSON response summarizing each thought tree and the final classification, following this format:

    ```json
    {{
        "Tree 1 - Emotional Analysis": "...",
        "Tree 2 - Behavioral Indicators": "...",
        "Tree 3 - Risk & Protective Factors": "...",
        "Label": "Final Classification"
    }}
    ```
    """

    return prompt
