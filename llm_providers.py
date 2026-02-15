"""
LLM Provider implementations for OpenAI, Claude, and Gemini.
"""

import os
import json
import time
import tempfile
import abc
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

from openai import OpenAI
from google import genai
from google.genai import types
import anthropic

import config

class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str, debug: bool = False):
        self.model = model
        self.debug = debug

    def debug_print(self, *args, **kwargs):
        """Print only if DEBUG mode is enabled."""
        if self.debug:
            print("[DEBUG]", *args, **kwargs)

    @abc.abstractmethod
    def test_deaths(self, deaths: List[Dict], **kwargs) -> List[Dict]:
        """
        Test the LLM's knowledge of deaths.
        Returns a list of result dictionaries.
        """
        pass

    def _parse_structured_response(self, raw_text: str) -> bool:
        """
        Parse a structured JSON response with {"answer": bool}.
        
        Args:
            raw_text: The string containing the JSON response.
            
        Returns:
            bool: True if the model knows about the death (answer=False), 
                  False if the model thinks they are alive (answer=True).
                  
        Raises:
            ValueError: If parsing fails or schema is incorrect.
        """
        try:
            # Clean up potential markdown formatting if present
            cleaned_text = raw_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            parsed = json.loads(cleaned_text)
            answer = parsed.get("answer")
            
            if answer is None:
                 # Fallback: try to find "answer" in keys if case mismatch
                 for k, v in parsed.items():
                     if k.lower() == "answer":
                         answer = v
                         break

            if not isinstance(answer, bool):
                raise ValueError(f"Expected boolean 'answer', got {type(answer).__name__}: {answer}")
            
            # answer=True means "alive" → doesn't know about death (knows_death=False)
            # answer=False means "dead" → knows about death (knows_death=True)
            return not answer
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to parse structured response: {e}")

    def _parse_structured_response_safe(self, raw_text: str) -> Tuple[Optional[bool], Optional[str]]:
        """
        Parse a structured JSON response safely.
        
        Returns:
            tuple: (knows_death, error_message). 
                   If successful, error_message is None.
                   If failed, knows_death is None.
        """
        try:
            return self._parse_structured_response(raw_text), None
        except ValueError as e:
            return None, str(e)


class OpenAIProvider(LLMProvider):
    """OpenAI Provider using sequential/flex API."""

    def __init__(self, model: str = config.DEFAULT_MODEL_OPENAI, debug: bool = False):
        super().__init__(model, debug)
        self.client = OpenAI() # Uses OPENAI_API_KEY environment variable

    def check_death_knowledge(self, name: str, description: str, max_tokens: int, reasoning_effort: str, temperature: float) -> Tuple[bool, str]:
        """
        Ask the LLM if a person has died.
        """
        prompt = config.PROMPT_TEMPLATE.format(name=name, description=description)
        current_max_tokens = max_tokens
        max_retries_tokens = config.MAX_TOKENS_REASONING_RETRY
        system_prompt = config.SYSTEM_PROMPT_REASONING if reasoning_effort != "none" else config.SYSTEM_PROMPT
        
        self.debug_print(f"Prompt: \"{prompt}\"")

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=current_max_tokens,
                    temperature=temperature,
                    service_tier="flex",
                    reasoning_effort=reasoning_effort,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "alive_check",
                            "strict": True,
                            "schema": config.STRUCTURED_SCHEMA
                        }
                    }
                )
                
                self.debug_print(f"Full response object: {response}")
                
                answer = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                
                self.debug_print(f"finish_reason: {finish_reason}")
                
                # Check if we got an empty response due to length limit with reasoning enabled
                if (not answer or answer.strip() == "") and finish_reason == "length" and reasoning_effort != "none":
                    if current_max_tokens < max_retries_tokens:
                        # Increase tokens and retry
                        new_max_tokens = min(current_max_tokens * 2, max_retries_tokens)
                        print(f"Empty response (finish_reason=length), retrying with max_tokens={new_max_tokens}...", end=" ")
                        current_max_tokens = new_max_tokens
                        continue
                    else:
                        print(f"Empty response even with max_tokens={current_max_tokens}")
                
                if answer:
                    answer = answer.strip()
                else:
                    answer = ""
                print(f"Answer: '{answer}'")
                return self._parse_structured_response(answer), answer
            
            except Exception as e:
                print(f"\n  Error querying for {name}: {e}")
                raise

    def test_deaths(self, deaths: List[Dict], delay: float = 0.1, reasoning: bool = False, **kwargs) -> Tuple[List[Dict], bool]:
        """
        Test the LLM's knowledge of deaths (OpenAI Flex - sequential).
        Returns (results, had_error) tuple.
        """
        results = []
        had_error = False
        
        max_tokens = config.MAX_TOKENS_LOW_REASONING if reasoning else config.MAX_TOKENS_OPENAI
        reasoning_effort = "low" if reasoning else "none"
        temperature = 1 if reasoning else config.TEMPERATURE
        system_prompt = config.SYSTEM_PROMPT_REASONING if reasoning else config.SYSTEM_PROMPT
        
        if self.debug:
            print("\n" + "=" * 60)
            print(f"[DEBUG] OpenAI SETTINGS")
            print("=" * 60)
            print(f"  model: {self.model}")
            print(f"  max_tokens: {max_tokens}")
            print(f"  reasoning_effort: {reasoning_effort}")
            print(f"  temperature: {temperature}")
            print(f"  service_tier: flex")
            print(f"  system_prompt: \"{system_prompt}\"")
            print(f"  prompt_template: {config.PROMPT_TEMPLATE}")
            print("=" * 60 + "\n")
        
        total = len(deaths)
        print(f"Testing {total} deaths with model: {self.model} (max_tokens={max_tokens}, reasoning_effort={reasoning_effort})")
        print("=" * 50)
        
        for i, death in enumerate(deaths):
            print(f"[{i+1}/{total}] {death['name']} (died {death['death_date']})...", end=" ")
            
            try:
                knows_death, response = self.check_death_knowledge(
                    death['name'], death.get('description', ''), max_tokens, reasoning_effort, temperature
                )
                
                results.append({
                    **death,
                    'llm_knows_death': knows_death,
                    'llm_response': response
                })
                
                if knows_death is True:
                    print("✓ Knows")
                elif knows_death is False:
                    print("✗ Doesn't know")
                else:
                    print("! Error")
            except Exception as e:
                print(f"\nFatal error at item {i+1}: {e}")
                had_error = True
                break
            
            if i < total - 1:
                time.sleep(delay)
        
        return results, had_error


class ClaudeBatchProvider(LLMProvider):
    """Claude Provider using Batch API."""

    def __init__(self, model: str = config.DEFAULT_MODEL_CLAUDE, debug: bool = False):
        super().__init__(model, debug)
        self.client = anthropic.Anthropic()

    def is_structured_supported(self) -> bool:
        """Check if a Claude model supports native structured outputs."""
        import re
        match = re.search(r'(\d+)[\.-](\d+)', self.model)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            return major > 4 or (major == 4 and minor >= 5)
        return False

    def _print_result(self, index: int, total: int, name: str, death_date: str, knows_death: bool, response: str = None):
        """Print a single test result."""
        prefix = f"[{index+1}/{total}] {name} (died {death_date})..."
        if response:
            prefix += f" Answer: {response}"
        
        if knows_death is True:
            print(f"{prefix} ✓ Knows")
        elif knows_death is False:
            print(f"{prefix} ✗ Doesn't know")
        else:
            print(f"{prefix} ! Error")

    def test_deaths(self, deaths: List[Dict], reasoning: bool = False, **kwargs) -> List[Dict]:
        """
        Test Claude's knowledge of deaths using the Batch API.
        """
        use_native_structured = self.is_structured_supported()
        
        temperature = 1 if reasoning else config.TEMPERATURE
        
        if reasoning:
            thinking_budget = 1024
            max_tokens = thinking_budget + config.MAX_TOKENS_CLAUDE
            thinking_config = {"type": "enabled", "budget_tokens": thinking_budget}
        else:
            max_tokens = config.MAX_TOKENS_CLAUDE
            thinking_config = {"type": "disabled"}
        
        if use_native_structured:
            max_tokens = max(max_tokens, 30)
        
        total = len(deaths)
        structured_mode = "native output_config" if use_native_structured else "prefilling+stop_sequence"
        print(f"Testing {total} deaths with model: {self.model} (using Batch API, structured={structured_mode}, max_tokens={max_tokens}, thinking={'enabled (1024)' if reasoning else 'disabled'})")
        print("=" * 50)
        
        system_prompt = config.SYSTEM_PROMPT_REASONING if reasoning else config.SYSTEM_PROMPT
        if not use_native_structured:
            system_prompt += ' Respond with a JSON object {"answer": true} if alive, {"answer": false} if dead.'
        
        if self.debug:
            print("\n" + "=" * 60)
            print(f"[DEBUG] Claude SETTINGS")
            print("=" * 60)
            print(f"  model: {self.model}")
            print(f"  max_tokens: {max_tokens}")
            print(f"  temperature: {temperature}")
            print(f"  thinking: {thinking_config}")
            print(f"  structured_mode: {structured_mode}")
            print(f"  system_prompt: \"{system_prompt}\"")
            print(f"  prompt_template: {config.PROMPT_TEMPLATE}")
            print("=" * 60 + "\n")
        
        batch_requests = []
        for i, death in enumerate(deaths):
            prompt = config.PROMPT_TEMPLATE.format(name=death['name'], description=death.get('description', ''))
            params = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "thinking": thinking_config
            }
            
            if use_native_structured:
                params["messages"] = [{"role": "user", "content": prompt}]
                params["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": config.STRUCTURED_SCHEMA
                    }
                }
            else:
                params["messages"] = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": '{"answer":'}
                ]
                params["stop_sequences"] = ["}"]
            
            batch_requests.append({
                "custom_id": str(i),
                "params": params
            })
            if i == 0:
                self.debug_print(f"Sample request (first): {batch_requests[0]}")
        
        print(f"Submitting batch of {len(batch_requests)} requests...")
        batch = self.client.messages.batches.create(requests=batch_requests)
        batch_id = batch.id
        print(f"Batch created: {batch_id}")
        
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            counts = batch.request_counts
            print(f"\rStatus: {status} | Succeeded: {counts.succeeded}/{total} | Processing: {counts.processing} | Errored: {counts.errored}".ljust(120), end="", flush=True)
            
            if status == "ended":
                print()
                break
            time.sleep(5)
        
        print("Retrieving results...")
        result_map = {}
        for entry in self.client.messages.batches.results(batch_id):
            custom_id = entry.custom_id
            self.debug_print(f"Result {custom_id}: {entry}")
            if entry.result.type == "succeeded":
                try:
                    raw_text = entry.result.message.content[0].text.strip()
                    
                    if use_native_structured:
                        knows_death, error = self._parse_structured_response_safe(raw_text)
                    else:
                        full_json = '{"answer":' + raw_text + '}'
                        knows_death, error = self._parse_structured_response_safe(full_json)
                        raw_text = full_json
                    
                    if error:
                        result_map[custom_id] = (None, f"{raw_text} (parse error: {error})")
                    else:
                        result_map[custom_id] = (knows_death, raw_text)
                except Exception as e:
                    result_map[custom_id] = (None, f"Error parsing response: {e}")
            else:
                result_map[custom_id] = (None, f"Error: {entry.result.type}")
        
        results = []
        for i, death in enumerate(deaths):
            knows_death, response = result_map.get(str(i), (None, "No result"))
            
            self.debug_print(f"Matching result {i} to {death['name']} (died {death['death_date']}): {response}")
            
            result = {
                **death,
                'llm_knows_death': knows_death,
                'llm_response': response
            }
            results.append(result)
            self._print_result(i, total, death['name'], death['death_date'], knows_death, response)
        
        return results


class GeminiBatchProvider(LLMProvider):
    """Gemini Provider using Batch API with file input."""

    def __init__(self, model: str = config.DEFAULT_MODEL_GEMINI, debug: bool = False):
        super().__init__(model, debug)
        self.client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))

    def _print_result(self, index: int, total: int, name: str, death_date: str, knows_death: bool, response: str = None):
        """Print a single test result."""
        prefix = f"[{index+1}/{total}] {name} (died {death_date})..."
        if response:
            prefix += f" Answer: {response}"
        
        if knows_death is True:
            print(f"{prefix} ✓ Knows")
        elif knows_death is False:
            print(f"{prefix} ✗ Doesn't know")
        else:
            print(f"{prefix} ! Error")

    def test_deaths(self, deaths: List[Dict], reasoning: bool = False, **kwargs) -> List[Dict]:
        """
        Test Gemini's knowledge of deaths using the Batch API with file-based input.
        """
        total = len(deaths)
        print(f"Testing {total} deaths with model: {self.model} (using Batch API with file input)")
        print("=" * 50)
        
        if reasoning or "gemini-3-pro" in self.model.lower():
            thinking_level = "low"
            max_tokens = config.MAX_TOKENS_LOW_REASONING
            temperature = 1
        else:
            thinking_level = "minimal"
            max_tokens = config.MAX_TOKENS_GEMINI_MINIMAL
            temperature = config.TEMPERATURE
        
        if self.debug:
            print("\n" + "=" * 60)
            print(f"[DEBUG] Gemini SETTINGS")
            print("=" * 60)
            print(f"  model: {self.model}")
            print(f"  max_output_tokens: {max_tokens}")
            print(f"  temperature: {temperature}")
            print(f"  thinking_level: {thinking_level}")
            print(f"  system_prompt: \"{config.SYSTEM_PROMPT_REASONING if thinking_level == 'low' else config.SYSTEM_PROMPT}\"")
            print(f"  prompt_template: {config.PROMPT_TEMPLATE}")
            print("=" * 60 + "\n")
        
        jsonl_file_path = tempfile.mktemp(suffix='.jsonl', prefix='gemini_batch_')
        
        with open(jsonl_file_path, 'w', encoding='utf-8') as f:
            for i, death in enumerate(deaths):
                prompt = config.PROMPT_TEMPLATE.format(name=death['name'], description=death.get('description', ''))
                request_obj = {
                    "key": f"request-{i}",
                    "request": {
                        "contents": [{
                            "parts": [{"text": prompt}],
                            "role": "user"
                        }],
                        "systemInstruction": {"parts": [{"text": config.SYSTEM_PROMPT_REASONING if thinking_level == 'low' else config.SYSTEM_PROMPT}]}
                    }
                }
                gen_cfg = {
                    "temperature": float(temperature),
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": config.STRUCTURED_SCHEMA
                }
                if thinking_level is not None:
                    gen_cfg["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}
                request_obj["request"]["generationConfig"] = gen_cfg
                f.write(json.dumps(request_obj) + '\n')
                if i == 0:
                    self.debug_print(f"Sample request (first): {request_obj}")
        
        print(f"Created JSONL file with {total} requests: {jsonl_file_path}")
        
        print("Uploading batch request file...")
        uploaded_file = self.client.files.upload(
            file=jsonl_file_path,
            config={'display_name': f'cutoff-test-{self.model}', 'mime_type': 'application/jsonl'}
        )
        print(f"Uploaded file: {uploaded_file.name}")
        
        print(f"Submitting batch job (thinking_level={thinking_level})...")
        batch_job = self.client.batches.create(
            model=self.model,
            src=uploaded_file.name,
            config={'display_name': f'cutoff-test-{self.model}'}
        )
        job_name = batch_job.name
        print(f"Batch created: {job_name}")
        
        completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
        while True:
            batch_job = self.client.batches.get(name=job_name)
            state = batch_job.state.name
            print(f"\rStatus: {state}".ljust(120), end="", flush=True)
            
            if state in completed_states:
                print()
                break
            time.sleep(5)
        
        if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
            print(f"Warning: Batch job ended with state: {batch_job.state.name}")
        
        print("Retrieving results...")
        result_map = {}
        
        if batch_job.dest and batch_job.dest.file_name:
            result_file_name = batch_job.dest.file_name
            print(f"Downloading results from: {result_file_name}")
            
            file_content_bytes = self.client.files.download(file=result_file_name)
            file_content = file_content_bytes.decode('utf-8')
            
            for line in file_content.splitlines():
                if not line.strip():
                    continue
                try:
                    parsed_response = json.loads(line)
                    key = parsed_response.get('key', '')
                    if key.startswith('request-'):
                        idx = int(key.split('-')[1])
                    else:
                        self.debug_print(f"Unexpected key format: {key}")
                        continue
                    
                    if 'response' in parsed_response and parsed_response['response']:
                        try:
                            candidates = parsed_response['response'].get('candidates', [])
                            if candidates:
                                parts = candidates[0].get('content', {}).get('parts', [])
                                answer = ''
                                for part in parts:
                                    if 'text' in part:
                                        answer = part['text'].strip()
                                        break
                                
                                knows_death, error = self._parse_structured_response_safe(answer)
                                if error:
                                    result_map[str(idx)] = (None, f"{answer} (parse error: {error})")
                                else:
                                    result_map[str(idx)] = (knows_death, answer)
                            else:
                                result_map[str(idx)] = (None, "Error: No candidates in response")
                        except Exception as e:
                            result_map[str(idx)] = (None, f"Error parsing response: {e}")
                    elif 'error' in parsed_response:
                        result_map[str(idx)] = (None, f"Error: {parsed_response['error']}")
                    else:
                        result_map[str(idx)] = (None, "Error: No response or error in result")
                        
                except json.JSONDecodeError as e:
                    self.debug_print(f"Failed to parse line: {line[:100]}... Error: {e}")
            
            try:
                self.client.files.delete(name=uploaded_file.name)
                self.debug_print(f"Deleted uploaded file: {uploaded_file.name}")
            except Exception as e:
                self.debug_print(f"Failed to delete uploaded file: {e}")
        
        try:
            os.remove(jsonl_file_path)
            self.debug_print(f"Deleted local temp file: {jsonl_file_path}")
        except Exception as e:
            self.debug_print(f"Failed to delete temp file: {e}")
        
        results = []
        for i, death in enumerate(deaths):
            knows_death, response = result_map.get(str(i), (None, "No result"))
            
            self.debug_print(f"Matching result {i} to {death['name']} (died {death['death_date']}): {response}")
            
            result = {
                **death,
                'llm_knows_death': knows_death,
                'llm_response': response
            }
            results.append(result)
            self._print_result(i, total, death['name'], death['death_date'], knows_death, response)
        
        return results
