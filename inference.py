import os
import sys
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

from env import HealthEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "no-key",
)

def llm_choose_action(heart_rate: int, temperature: float) -> int:
    """Use OpenAI client to decide which action to take."""
    prompt = (
        f"You are a health monitoring AI. "
        f"Current vitals: heart_rate={heart_rate}, temperature={temperature}. "
        f"Choose one action:\n"
        f"  0 = do_nothing (vitals are normal)\n"
        f"  1 = send_warning (vitals are mildly concerning)\n"
        f"  2 = emergency_alert (vitals are critically abnormal)\n"
        f"Reply with ONLY the single digit 0, 1, or 2."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        action = int(text[0])
        if action not in (0, 1, 2):
            raise ValueError(f"Out of range: {action}")
        return action
    except Exception:
        # Fallback rule-based logic if LLM call fails
        if heart_rate > 120 or temperature > 39:
            return 2
        elif heart_rate > 100 or temperature > 38:
            return 1
        return 0


env = HealthEnv()

print("[START]")

state = env.reset()
done = False
step_count = 0
total_reward = 0.0
print(f"[STEP] step={step_count} state={state} action=None reward=0.0 done={done}")

while not done and step_count < 20:
    hr   = state["heart_rate"]
    temp = state["temperature"]

    action = llm_choose_action(hr, temp)

    next_state, reward, done, _ = env.step(action)

    step_count    += 1
    total_reward  += reward

    print(f"[STEP] step={step_count} state={next_state} action={action} reward={reward} done={done}")

    state = next_state


print(f"[END] total_steps={step_count} total_reward={round(total_reward, 4)}")