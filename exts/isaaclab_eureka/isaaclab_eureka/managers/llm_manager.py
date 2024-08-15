# Copyright (c) 2022-2024, The IsaacLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import openai
import re
import requests
import time


class LLMManager:
    """Manager to interface with the LLM API.
    
    This class is responsible for interfacing with the LLM API to generate rewards.
    It establishes a connection either to native OpenAI API, or to the Azure OpenAI API.
    
    """

    def __init__(self, gpt_model: str, num_suggestions: int, temperature: float, system_prompt: str):
        """Initialize the LLMManager

        The Openai API relies on the following environment variables:
        - For the native OpenAI API, the environment variable OPENAI_API_KEY must be set.
        - For the Azure OpenAI API, the environment variables AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set.

        Args:
            gpt_model: The model to use for the LLM API
            num_suggestions: The number of independent suggestions to generate
            temperature: The temperature to use for the LLM API
            system_prompt: The system prompt to provide to the LLM API
        """

        self._gpt_model = gpt_model
        self._num_suggestions = num_suggestions
        self._temperature = temperature
        self._prompts = [{"role": "system", "content": system_prompt}]

        if "AZURE_OPENAI_API_KEY" in os.environ:
            self._client = openai.AzureOpenAI(api_version="2024-02-01")
        elif "OPENAI_API_KEY" in os.environ:
            self._client = openai.OpenAI()
        else:
            raise RuntimeError("No Openai API key found in environment variables")

    def extract_code_from_response(self, response: str) -> str:
        """Extract the code component from the LLM response

        If the response contains a code block of the form "```python ... ```", extract the code block from the response.
        Otherwise, return the entire response of the LLM.
        
        Args:
            response: The response from the LLM API
        """
        pattern = r"```python(.*?)```"
        result = re.findall(pattern, response, re.DOTALL)
        code_string = response
        if result is not None and len(result) > 0:
            code_string = result[-1]
            # Remove leading newline characters
            code_string = code_string.lstrip('\n')
        return code_string

    def prompt(self, user_prompt: str, assistant_prompt: str = None) -> list[str]:
        """Call the LLM API to collect responses
        
        Args:
            user_prompt: The user prompt to provide to the LLM API
            assistant_prompt: The assistant prompt to provide to the LLM API

        Returns:
            A dictionary containing the reward strings and raw outputs from the LLM

        Raises:
            Exception: If there is an error with the LLM API
        """
        if assistant_prompt is not None:
            self._prompts.append({"role": "assistant", "content": assistant_prompt})
        self._prompts.append({"role": "user", "content": user_prompt})

        # The official Eureka code only keeps the last round of feedback
        if len(self._prompts) == 6:
            self._prompts.pop(2)
            self._prompts.pop(2)

        try:
            responses = self._client.chat.completions.create(
                model=self._gpt_model,
                messages=self._prompts,
                temperature=self._temperature,
                n=self._num_suggestions,
            )
        except Exception as e:
            raise RuntimeError(f"An error occurred while prompting the LLM") from e

        raw_outputs = [response.message.content for response in responses.choices]
        reward_strings = [self.extract_code_from_response(raw_output) for raw_output in raw_outputs]
        return {"reward_strings": reward_strings, "raw_outputs": raw_outputs}
