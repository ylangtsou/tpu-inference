# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import asdict
from unittest.mock import patch

import pytest
from vllm import LLM, EngineArgs, SamplingParams

from tpu_inference.core.core_tpu import DisaggEngineCore, DisaggEngineCoreProc


@pytest.fixture
def test_prompts():
    """Simple test prompts for disaggregated serving testing."""
    return [
        "Hello, my name is",
        "The capital of France is",
        "The colors of the rainbow are",
        "The future of AI is",
        "The president of the United States is",
        "How many players are on a standard soccer team on the field at one time?",
        "In Greek mythology, who is the god of the sea?",
        "In what year did the Titanic sink?",
        "In which museum is the Mona Lisa displayed?",
        "Mount Everest is located in which mountain range?",
        "What ancient empire was ruled by Julius Caesar?",
        "What are the four fundamental forces of nature?",
        'What does "CPU" stand for?',
        'What does "HTML" stand for?',
        "What is the capital of Australia?",
        "What is the chemical symbol for gold?",
        "What is the currency of Switzerland?",
        "What is the distance from the Earth to the Sun called?",
        "What is the freezing point of water in Celsius?",
        "What is the hardest known natural substance on Earth?",
        "What is the largest planet in our solar system?",
        "What is the longest river in the world?",
        "What is the main function of the kidneys in the human body?",
        "What is the main ingredient in guacamole?",
        "What is the most spoken language in the world by number of native speakers?",
        "What is the process by which plants use sunlight to create food?",
        "Which country is known as the Land of the Rising Sun?",
        "Who developed the theory of general relativity?",
        'Who directed the original "Star Wars" trilogy?',
        "Who is credited with inventing the telephone?",
        "Who painted the ceiling of the Sistine Chapel?",
        "Who was the first female Prime Minister of the United Kingdom?",
        "Who was the first person to walk on the moon?",
        "Who wrote the American Declaration of Independence?",
        'Who wrote the novel "Pride and Prejudice"?',
    ]


@pytest.fixture
def sampling_params():
    """Standard sampling parameters for testing."""
    return SamplingParams(
        temperature=0.0,
        max_tokens=128,
        ignore_eos=True,
    )


def test_disaggregated_serving(test_prompts, sampling_params):
    """
    Test disaggregated serving end-to-end.

    Equivalent to:
    PREFILL_SLICES=4 DECODE_SLICES=4 python examples/offline_inference.py \
        --model=meta-llama/Meta-Llama-3.1-8B-Instruct --task=generate \
        --max_model_len=2048 --tensor_parallel_size 4
    """
    # Set environment variables for disaggregated serving
    # Using 4 slices for prefill and 4 for decode as requested
    # Note: The user example used PREFILL_SLICES=4 DECODE_SLICES=4
    # But usually slices are specified as "2x2" or similar if they are TPU topology.
    # However, disagg_utils.py _parse_slices handles "4" as well (1D).
    # We will stick to the user's example values.

    # We need to mock the environment variables for this test
    with patch.dict(
            os.environ, {
                "PREFILL_SLICES": "4",
                "DECODE_SLICES": "4",
                "SKIP_JAX_PRECOMPILE": "1",
                "VLLM_XLA_CHECK_RECOMPILATION": "0"
            }):
        # Patch the EngineCore classes to use Disagg versions
        with patch("vllm.v1.engine.core.EngineCore", DisaggEngineCore), \
             patch("vllm.v1.engine.core.EngineCoreProc", DisaggEngineCoreProc):

            model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

            engine_args = EngineArgs(
                model=model_name,
                max_model_len=2048,
                tensor_parallel_size=4,
                gpu_memory_utilization=0.90,
                enforce_eager=False,
            )

            llm = LLM(**asdict(engine_args))

            try:
                outputs = llm.generate(test_prompts, sampling_params)

                # Verify outputs
                assert len(outputs) == len(test_prompts)
                for output in outputs:
                    assert len(output.outputs) > 0
                    assert len(output.outputs[0].text.strip()) > 0
                    print(f"Prompt: {output.prompt!r}")
                    print(f"Generated: {output.outputs[0].text!r}")

            finally:
                # Clean up if needed, though LLM destructor usually handles it
                pass
