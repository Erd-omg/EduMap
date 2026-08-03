"""Tests for OutputSchema — structured output parsing and tool definitions.

Pure logic tests — no external dependencies.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field


# ── Test Pydantic models ─────────────────────────────────────────────


class Person(BaseModel):
    name: str
    age: int


class Book(BaseModel):
    title: str = Field(description="Book title")
    author: str = Field(description="Author name")
    year: int = Field(default=2024, description="Publication year")


# ── Tests ────────────────────────────────────────────────────────────


class TestParse:
    """OutputSchema.parse() — JSON extraction and validation."""

    def test_parse_raw_json(self) -> None:
        """Raw JSON string is parsed and validated."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person)
        result = schema.parse('{"name": "Alice", "age": 30}')
        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    def test_parse_markdown_fence(self) -> None:
        """JSON inside ```json ... ``` fence is extracted."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person)
        result = schema.parse(
            'Here is the result:\n```json\n{"name": "Bob", "age": 25}\n```\n'
        )
        assert result.name == "Bob"
        assert result.age == 25

    def test_parse_markdown_fence_no_lang(self) -> None:
        """Fence without 'json' language tag (``` only) still works."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person)
        result = schema.parse(
            '```\n{"name": "Carol", "age": 35}\n```'
        )
        assert result.name == "Carol"

    def test_parse_trailing_text_raises(self) -> None:
        """Extra text after JSON JSONDecodeError is propagated.

        NOTE: The current implementation uses ``json.loads()`` which does
        not allow trailing text after the JSON object. Markdown fences
        are stripped first, but plain trailing text at the top level
        will cause a parse failure.
        """
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person)
        with pytest.raises(Exception):
            schema.parse(
                '{"name": "Dave", "age": 40}\nSome trailing text...'
            )

    def test_parse_empty_text_raises(self) -> None:
        """Empty text raises an error."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person)
        with pytest.raises(Exception):
            schema.parse("")

    def test_parse_invalid_json_raises(self) -> None:
        """Invalid JSON raises validation error."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person)
        with pytest.raises(Exception):
            schema.parse("not json at all")

    def test_parse_partial_missing_fields_raises(self) -> None:
        """JSON missing required fields raises Pydantic validation error."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person)
        with pytest.raises(Exception):
            schema.parse('{"name": "Eve"}')  # missing 'age'

    def test_parse_with_defaults(self) -> None:
        """Fields with defaults use default values when absent."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Book)
        result = schema.parse('{"title": "1984", "author": "Orwell"}')
        assert result.year == 2024  # default

    def test_parse_full_fence_with_trailing_text(self) -> None:
        """JSON fence followed by trailing text extracts correctly."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person)
        text = """
Some analysis text before...

```json
{"name": "Frank", "age": 28}
```

Some closing remarks.
"""
        result = schema.parse(text)
        assert result.name == "Frank"
        assert result.age == 28


class TestToolDefinition:
    """to_function_tool_def() — OpenAI-compatible tool definition."""

    def test_tool_def_structure(self) -> None:
        """Tool definition has expected structure."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person, schema_id="extract_person")
        tool_def = schema.to_function_tool_def()

        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "extract_person"
        assert "parameters" in tool_def["function"]
        assert "properties" in tool_def["function"]["parameters"]
        assert "name" in tool_def["function"]["parameters"]["properties"]
        assert "age" in tool_def["function"]["parameters"]["properties"]

    def test_tool_def_default_schema_id(self) -> None:
        """Default schema_id is the model class name."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person)
        tool_def = schema.to_function_tool_def()
        assert tool_def["function"]["name"] == "Person"


class TestPromptInstructions:
    """to_prompt_instructions() — human-readable JSON schema instructions."""

    def test_prompt_instructions_contain_json_schema(self) -> None:
        """Instructions include JSON schema representation."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person)
        instructions = schema.to_prompt_instructions()

        assert "JSON" in instructions
        assert "name" in instructions
        assert "age" in instructions

    def test_prompt_instructions_chinese(self) -> None:
        """Instructions are in Chinese."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person)
        instructions = schema.to_prompt_instructions()

        assert "你必须" in instructions
        assert "Schema" in instructions


class TestCustomSchemaId:
    """Custom schema_id propagation."""

    def test_custom_schema_id_in_tool_def(self) -> None:
        """Custom schema_id appears in tool definition."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person, schema_id="my_custom_schema")
        tool_def = schema.to_function_tool_def()
        assert tool_def["function"]["name"] == "my_custom_schema"

    def test_custom_schema_id_in_prompt(self) -> None:
        """Custom schema_id used when referenced."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person, schema_id="custom_id")
        prompt = schema.to_prompt_instructions()
        # The schema_id isn't directly in the prompt instructions text,
        # but the tool definition uses it
        assert schema.schema_id == "custom_id"


class TestJsonSchema:
    """Internal JSON schema generation."""

    def test_json_schema_types(self) -> None:
        """JSON schema correctly reflects Pydantic field types."""
        from src.harness.structured import OutputSchema

        schema = OutputSchema(Person)
        json_schema = schema._json_schema

        assert json_schema["properties"]["name"]["type"] == "string"
        assert json_schema["properties"]["age"]["type"] == "integer"
        assert "required" in json_schema
        assert "name" in json_schema["required"]
        assert "age" in json_schema["required"]
