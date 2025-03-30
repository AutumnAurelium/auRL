import openai
from dotenv import load_dotenv
import os
import re

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")

SUBJECTIVE_REWARD_PROMPT = """You will receive a poem written by an LLM. Please evaluate the poem based on the below rubric:

# LLM Poetry Evaluation Rubric

## Technical Elements (30%)

### Structure and Form (10%)
- **5**: Masterful command of the chosen form with purposeful deviations that enhance meaning
- **4**: Strong adherence to chosen form with occasional innovative variations
- **3**: Adequate adherence to basic structural elements of chosen form
- **2**: Inconsistent structure with limited understanding of the chosen form
- **1**: No discernible structure or inappropriate use of form

### Sonic Qualities (10%)
- **5**: Exceptional use of sound devices (rhyme, rhythm, alliteration, assonance) that significantly enhance meaning
- **4**: Effective use of sound devices that generally enhance the poem
- **3**: Basic use of sound devices that occasionally support the poem
- **2**: Limited or inconsistent use of sound devices
- **1**: No discernible sound patterns or distracting/forced sonic elements

### Technical Skill (10%)
- **5**: Exceptional command of language, meter, and technical elements
- **4**: Strong technical execution with minor imperfections
- **3**: Competent technical execution with some noticeable issues
- **2**: Significant technical weaknesses that undermine the poem
- **1**: Major technical problems throughout

## Content Elements (40%)

### Imagery and Sensory Detail (10%)
- **5**: Vivid, original imagery that creates powerful sensory experiences and deepens meaning
- **4**: Strong imagery that effectively engages multiple senses
- **3**: Adequate imagery that occasionally engages the senses
- **2**: Limited or clichéd imagery with minimal sensory engagement
- **1**: Vague, confusing, or absent imagery

### Meaning and Depth (10%)
- **5**: Profound insights and layered meanings that invite multiple interpretations
- **4**: Meaningful content with some depth and subtlety
- **3**: Clear meaning but limited depth or complexity
- **2**: Superficial or underdeveloped meaning
- **1**: No discernible meaning or incoherent content

### Emotional Resonance (10%)
- **5**: Genuine, nuanced emotional impact that lingers beyond reading
- **4**: Authentic emotional qualities that engage the reader
- **3**: Recognizable emotional elements but limited impact
- **2**: Forced or superficial emotional content
- **1**: No emotional engagement or inappropriate emotional tone

### Originality and Voice (10%)
- **5**: Highly distinctive voice with innovative approaches to subject/language
- **4**: Clear personal voice with fresh perspectives
- **3**: Recognizable voice with some original elements
- **2**: Derivative approach with minimal unique qualities
- **1**: Entirely conventional with no distinctive voice

## Cohesion Elements (30%)

### Unity and Coherence (10%)
- **5**: Perfect integration of all elements serving a unified purpose
- **4**: Strong coherence with elements generally working together
- **3**: Basic coherence with occasional disconnected elements
- **2**: Limited coherence with frequent disconnects
- **1**: Disjointed elements with no clear relationship

### Economy and Precision (10%)
- **5**: Every word essential, with perfect precision and no excess
- **4**: Generally economical with few unnecessary elements
- **3**: Mostly appropriate word choice with some excess
- **2**: Significant verbosity or imprecision
- **1**: Highly inefficient language throughout

### Progression and Movement (10%)
- **5**: Masterful development that creates compelling momentum
- **4**: Effective progression that maintains interest
- **3**: Basic movement with logical sequencing
- **2**: Stagnant or jumbled progression
- **1**: Random or contradictory movement

## Overall Assessment

### Holistic Rating (100-point scale)
- **90-100**: Exceptional poetry demonstrating mastery across all elements
- **80-89**: Strong poetry with notable strengths and minor weaknesses
- **70-79**: Competent poetry meeting basic expectations
- **60-69**: Developing poetry with significant room for improvement
- **Below 60**: Needs fundamental reconsideration

Evalaute the poem, and then reply with the score in an XML tag, like this:

<score>93</score>"""

poem_topics = [
    "dawn in the mountains",
    "urban solitude",
    "first love",
    "grief and healing",
    "climate change",
    "artificial intelligence",
    "ocean depths",
    "childhood memories",
    "social justice",
    "interstellar travel",
    "seasons changing",
    "dreams and nightmares",
    "ancient ruins",
    "identity and belonging",
    "the passage of time",
    "pandemic isolation",
    "forests at night",
    "digital relationships",
    "migration journeys",
    "forgotten languages",
    "parenthood",
    "musical instruments",
    "war and peace",
    "technological dependence",
    "lost civilizations",
    "human resilience",
    "cosmic insignificance",
    "garden growth",
    "cultural heritage",
    "everyday objects",
    "animal consciousness",
    "city architecture",
    "religious faith",
    "scientific discovery",
    "river journeys",
    "mental health struggles",
    "artistic inspiration",
    "desert landscapes",
    "lunar reflections",
    "human connection",
    "environmental destruction",
    "technological utopia",
    "mythological creatures",
    "culinary experiences",
    "political upheaval",
    "aging process",
    "quantum physics",
    "lost friendship",
    "urban wildlife",
    "forgotten histories"
]

def subjective_reward(prompts: list[str], completions: list[str]):
    rewards = []
    for completion in completions:
        poem = completion[-1]["content"]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SUBJECTIVE_REWARD_PROMPT},
                {"role": "user", "content": f"{poem}"}
            ]
        )
        
        evaluation = response.choices[0].message.content
        match = re.search(r"<score>(.*?)</score>", evaluation)
        if match:
            score = float(match.group(1))
        else:
            score = 0.0
        rewards.append(score)
        
    return rewards



