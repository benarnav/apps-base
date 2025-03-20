import os
import time
from datetime import datetime

from dotenv import load_dotenv

from inspect_ai import eval
from inspect_ai.model import get_model, GenerateConfig

import bigcodebench


models_names = [
    "together/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "openai/deepseek-r1-distill-qwen-32b",
]
models = []
model_config = GenerateConfig(
    temperature=0.7,
    top_p=0.95,
    max_tokens=16000,
)


for model in models_names:
    models.append(
        get_model(
            model=model,
            config=model_config,
            base_url="https://api.groq.com/openai/v1" if "32" in model else None,
            api_key=(
                os.getenv("OPENAI_API_KEY")
                if "32" in model
                else os.getenv("TOGETHER_API_KEY")
            ),
        )
    )


def main():
    start_time = time.time()

    log_directory = f"logs/logs_appsfull_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_directory, exist_ok=True)

    task = bigcodebench(
        num_epochs=2,
        version="v0.1.4",
    )

    eval(
        tasks=task,
        model=models,
        log_dir=log_directory,
        limit=20,
    )

    end_time = time.time()
    total_time = end_time - start_time
    print(
        f"Total time taken: {int((total_time) // 3600)}h {int(((total_time) % 3600) // 60)}m {int((total_time) % 60)}s"
    )


if __name__ == "__main__":
    main()
