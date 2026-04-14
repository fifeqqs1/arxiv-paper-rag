import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from langchain_ollama import ChatOllama
from src.config import Settings
from src.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError
from src.schemas.ollama import RAGResponse
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama local LLM service."""

    def __init__(self, settings: Settings):
        """Initialize Ollama client with settings."""
        self.settings = settings
        self.base_url = settings.ollama_host
        self.timeout = httpx.Timeout(float(settings.ollama_timeout))
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()
        self.qwen_base_url = settings.qwen_base_url.rstrip("/")
        self.qwen_api_key = settings.qwen_api_key.strip()

    def _resolve_provider(self, provider: Optional[str] = None) -> str:
        resolved = self.settings.resolve_llm_provider(provider)
        if resolved == "qwen_api" and not self.qwen_api_key:
            logger.warning("Qwen API provider requested but QWEN_API_KEY is empty, falling back to Ollama")
            return "ollama"
        return resolved

    def _resolve_model(self, provider: Optional[str], model: str) -> str:
        return self.settings.resolve_llm_model(provider, model)

    def _qwen_headers(self) -> Dict[str, str]:
        if not self.qwen_api_key:
            raise OllamaConnectionError("QWEN_API_KEY is not configured")
        return {
            "Authorization": f"Bearer {self.qwen_api_key}",
            "Content-Type": "application/json",
        }

    def _build_qwen_payload(self, model: str, prompt: str, stream: bool, **kwargs) -> Dict[str, Any]:
        allowed_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {"temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "stop"}
        }
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            **allowed_kwargs,
        }

    @staticmethod
    def _extract_qwen_text(message_content: Any) -> str:
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            parts = []
            for item in message_content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "".join(parts)
        return str(message_content or "")

    async def _qwen_health_check(self) -> Dict[str, Any]:
        headers = self._qwen_headers()
        models_url = f"{self.qwen_base_url}/models"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(models_url, headers=headers)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "message": "Qwen API is reachable",
                    "provider": "qwen_api",
                }

            # Fallback to a minimal generation probe if the models endpoint is unsupported.
            probe_response = await client.post(
                f"{self.qwen_base_url}/chat/completions",
                headers=headers,
                json=self._build_qwen_payload(
                    model=self.settings.qwen_model_name,
                    prompt="Reply with ok.",
                    stream=False,
                    temperature=0,
                    max_tokens=8,
                ),
            )
            if probe_response.status_code == 200:
                return {
                    "status": "healthy",
                    "message": "Qwen API is reachable",
                    "provider": "qwen_api",
                }
            raise OllamaException(f"Qwen API health check failed: {probe_response.status_code}")

    async def _qwen_list_models(self) -> List[Dict[str, Any]]:
        headers = self._qwen_headers()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.qwen_base_url}/models", headers=headers)
            if response.status_code != 200:
                raise OllamaException(f"Failed to list Qwen models: {response.status_code}")
            data = response.json().get("data", [])
            return [{"name": item.get("id", ""), "model": item.get("id", "")} for item in data if item.get("id")]

    async def _qwen_generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        headers = self._qwen_headers()
        payload = self._build_qwen_payload(model=model, prompt=prompt, stream=stream, **kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.qwen_base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            if response.status_code != 200:
                raise OllamaException(f"Qwen generation failed: {response.status_code} {response.text[:500]}")

            result = response.json()
            choice = (result.get("choices") or [{}])[0]
            message = choice.get("message", {})
            content = self._extract_qwen_text(message.get("content", ""))

            usage = result.get("usage", {})
            usage_metadata = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            usage_metadata = {k: v for k, v in usage_metadata.items() if v is not None}

            return {
                "response": content,
                "done": True,
                "done_reason": choice.get("finish_reason"),
                "usage_metadata": usage_metadata,
            }

    async def _qwen_generate_stream(self, model: str, prompt: str, **kwargs):
        headers = self._qwen_headers()
        payload = self._build_qwen_payload(model=model, prompt=prompt, stream=True, **kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.qwen_base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise OllamaException(f"Qwen streaming generation failed: {response.status_code} {body[:500]!r}")

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:].strip()
                    if not data_str:
                        continue
                    if data_str == "[DONE]":
                        yield {"done": True}
                        break

                    try:
                        payload = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse Qwen streaming chunk: {data_str}")
                        continue

                    choice = (payload.get("choices") or [{}])[0]
                    delta = choice.get("delta", {})
                    text_chunk = self._extract_qwen_text(delta.get("content", ""))
                    if text_chunk:
                        yield {"response": text_chunk}

                    if choice.get("finish_reason"):
                        yield {"done": True, "done_reason": choice.get("finish_reason")}
                        break

    def get_langchain_model(self, model: str, temperature: float = 0.0, **kwargs) -> ChatOllama:
        """Return a LangChain-compatible Ollama chat model for agentic workflows."""
        if self._resolve_provider() != "ollama":
            raise OllamaException("LangChain model adapter is only available for the Ollama provider")
        return ChatOllama(
            base_url=self.base_url,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Ollama service is healthy and responding.

        Returns:
            Dictionary with health status information
        """
        try:
            if self._resolve_provider() == "qwen_api":
                return await self._qwen_health_check()

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Check version endpoint for health
                response = await client.get(f"{self.base_url}/api/version")

                if response.status_code == 200:
                    version_data = response.json()
                    return {
                        "status": "healthy",
                        "message": "Ollama service is running",
                        "version": version_data.get("version", "unknown"),
                    }
                else:
                    raise OllamaException(f"Ollama returned status {response.status_code}")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Ollama health check failed: {str(e)}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models.

        Returns:
            List of model information dictionaries
        """
        try:
            if self._resolve_provider() == "qwen_api":
                return await self._qwen_list_models()

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")

                if response.status_code == 200:
                    data = response.json()
                    return data.get("models", [])
                else:
                    raise OllamaException(f"Failed to list models: {response.status_code}")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error listing models: {e}")

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        provider: Optional[str] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate text using specified model.

        Args:
            model: Model name to use
            prompt: Input prompt for generation
            stream: Whether to stream response
            **kwargs: Additional generation parameters

        Returns:
            Response dictionary with added usage_metadata field containing:
                - prompt_tokens: Number of tokens in the prompt
                - completion_tokens: Number of tokens in the completion
                - total_tokens: Total tokens used
                - latency_ms: Generation latency in milliseconds
        """
        try:
            resolved_provider = self._resolve_provider(provider)
            resolved_model = self._resolve_model(resolved_provider, model)

            if resolved_provider == "qwen_api":
                return await self._qwen_generate(model=resolved_model, prompt=prompt, stream=stream, **kwargs)

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                data = {"model": resolved_model, "prompt": prompt, "stream": stream, **kwargs}

                logger.info(
                    f"Sending request to Ollama: model={resolved_model}, stream={stream}, extra_params={kwargs}"
                )
                response = await client.post(f"{self.base_url}/api/generate", json=data)

                if response.status_code == 200:
                    result = response.json()

                    # Parse Ollama usage metadata and convert to Langfuse-compatible format
                    usage_metadata = {}

                    # Ollama returns these fields in the response
                    if "prompt_eval_count" in result:
                        usage_metadata["prompt_tokens"] = result.get("prompt_eval_count", 0)
                    if "eval_count" in result:
                        usage_metadata["completion_tokens"] = result.get("eval_count", 0)

                    # Calculate total tokens
                    if usage_metadata:
                        usage_metadata["total_tokens"] = (
                            usage_metadata.get("prompt_tokens", 0) +
                            usage_metadata.get("completion_tokens", 0)
                        )

                    # Parse timing information (convert nanoseconds to milliseconds)
                    if "total_duration" in result:
                        # Ollama returns duration in nanoseconds
                        usage_metadata["latency_ms"] = round(result["total_duration"] / 1_000_000, 2)

                    # Add timing breakdown if available
                    if "prompt_eval_duration" in result:
                        usage_metadata["prompt_eval_duration_ms"] = round(result["prompt_eval_duration"] / 1_000_000, 2)
                    if "eval_duration" in result:
                        usage_metadata["eval_duration_ms"] = round(result["eval_duration"] / 1_000_000, 2)

                    # Attach usage metadata to the response
                    result["usage_metadata"] = usage_metadata

                    logger.debug(f"Usage metadata: {usage_metadata}")

                    return result
                else:
                    raise OllamaException(f"Generation failed: {response.status_code}")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error generating with Ollama: {e}")

    async def generate_stream(self, model: str, prompt: str, provider: Optional[str] = None, **kwargs):
        """
        Generate text with streaming response.

        Args:
            model: Model name to use
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters

        Yields:
            JSON chunks from streaming response
        """
        try:
            resolved_provider = self._resolve_provider(provider)
            resolved_model = self._resolve_model(resolved_provider, model)

            if resolved_provider == "qwen_api":
                async for chunk in self._qwen_generate_stream(model=resolved_model, prompt=prompt, **kwargs):
                    yield chunk
                return

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                data = {"model": resolved_model, "prompt": prompt, "stream": True, **kwargs}

                logger.info(f"Starting streaming generation: model={resolved_model}")

                async with client.stream("POST", f"{self.base_url}/api/generate", json=data) as response:
                    if response.status_code != 200:
                        raise OllamaException(f"Streaming generation failed: {response.status_code}")

                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                yield chunk
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse streaming chunk: {line}")
                                continue

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error in streaming generation: {e}")

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "qwen2.5:7b",
        provider: Optional[str] = None,
        use_structured_output: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a RAG answer using retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved document chunks with metadata
            model: Model to use for generation
            use_structured_output: Whether to use Ollama's structured output feature

        Returns:
            Dictionary with answer, sources, confidence, and citations
        """
        try:
            if use_structured_output:
                # Use structured output with Pydantic model
                prompt_data = self.prompt_builder.create_structured_prompt(query, chunks)

                # Generate with structured format
                response = await self.generate(
                    model=model,
                    prompt=prompt_data["prompt"],
                    provider=provider,
                    temperature=0.7,
                    top_p=0.9,
                    format=prompt_data["format"],
                )
            else:
                # Fallback to plain text mode
                prompt = self.prompt_builder.create_rag_prompt(query, chunks)

                # Generate without format restrictions
                response = await self.generate(
                    model=model,
                    prompt=prompt,
                    provider=provider,
                    temperature=0.7,
                    top_p=0.9,
                )

            if response and "response" in response:
                answer_text = response["response"]
                logger.debug(f"Raw LLM response: {answer_text[:500]}")

                if use_structured_output:
                    # Try to parse structured response if enabled
                    parsed_response = self.response_parser.parse_structured_response(answer_text)
                    logger.debug(f"Parsed response: {parsed_response}")
                    return parsed_response
                else:
                    # For plain text response, build simple response structure
                    sources = []
                    seen_urls = set()
                    for chunk in chunks:
                        arxiv_id = chunk.get("arxiv_id")
                        if arxiv_id:
                            arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                            pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                            if pdf_url not in seen_urls:
                                sources.append(pdf_url)
                                seen_urls.add(pdf_url)

                    citations = list(set(chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")))

                    return {
                        "answer": answer_text,
                        "sources": sources,
                        "confidence": "medium",
                        "citations": citations[:5],
                    }
            else:
                raise OllamaException("No response generated from Ollama")

        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            raise OllamaException(f"Failed to generate RAG answer: {e}")

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "qwen2.5:7b",
        provider: Optional[str] = None,
    ):
        """
        Generate a streaming RAG answer using retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved document chunks with metadata
            model: Model to use for generation

        Yields:
            Streaming response chunks with partial answers
        """
        try:
            # Create prompt for streaming (simpler than structured)
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            # Stream the response
            async for chunk in self.generate_stream(
                model=model,
                prompt=prompt,
                provider=provider,
                temperature=0.7,
                top_p=0.9,
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Error generating streaming RAG answer: {e}")
            raise OllamaException(f"Failed to generate streaming RAG answer: {e}")
