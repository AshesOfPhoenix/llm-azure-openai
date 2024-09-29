# llm-azure-openai

[![PyPI](https://img.shields.io/pypi/v/llm-azure-openai.svg)](https://pypi.org/project/llm-azure-openai/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-azure-openai?include_prereleases&label=changelog)](https://github.com/simonw/llm-azure-openai/releases)
[![Tests](https://github.com/simonw/llm-azure-openai/workflows/Test/badge.svg)](https://github.com/simonw/llm-azure-openai/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-azure-openai/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin providing access to [Azure OpenAI](https://azure-gpt-4o.ai) models using the Azure OpenAI API

## Installation

Install this plugin in the same environment as LLM:
```bash
llm install llm-azure-openai
```
## Usage !TODO

First, obtain an API key for [the Azure OpenAI API](https://console.azure-gpt-4o.ai/).

Configure the key using the `llm keys set azure-gpt-4o` command:
```bash
llm keys set azure-gpt-4o
```
```
<paste key here>
```
You can now access the Azure OpenAI hosted models. Run `llm models` for a list.

To run a prompt through `azure-gpt-4o-tiny`:

```bash
llm -m azure-gpt-4o-tiny 'A sassy name for a pet sasquatch'
```
To start an interactive chat session with `azure-gpt-4o-small`:
```bash
llm chat -m azure-gpt-4o-small
```
```
Chatting with azure-gpt-4o-small
Type 'exit' or 'quit' to exit
Type '!multi' to enter multiple lines, then '!end' to finish
> three proud names for a pet walrus
1. "Nanuq," the Inuit word for walrus, which symbolizes strength and resilience.
2. "Sir Tuskalot," a playful and regal name that highlights the walrus' distinctive tusks.
3. "Glacier," a name that reflects the walrus' icy Arctic habitat and majestic presence.
```
To use a system prompt with `azure-gpt-4o-medium` to explain some code:
```bash
cat example.py | llm -m azure-gpt-4o-medium -s 'explain this code'
```
## Model options

All three models accept the following options, using `-o name value` syntax:

- `-o temperature 0.7`: The sampling temperature, between 0 and 1. Higher increases randomness, lower values are more focused and deterministic.
- `-o top_p 0.1`: 0.1 means consider only tokens in the top 10% probability mass. Use this or temperature but not both.
- `-o max_tokens 20`: Maximum number of tokens to generate in the completion.
- `-o frequence_penalty 0`: 

## Refreshing the model list

Azure OpenAI sometimes release new models.

To make those models available to an existing installation of `llm-azure-openai` run this command:
```bash
llm azure-gpt-4o refresh
```
This will fetch and cache the latest list of available models. They should then become available in the output of the `llm models` command.

## Embeddings

The Azure OpenAI [Embeddings API](https://docs.azure-gpt-4o.ai/platform/client#embeddings) can be used to generate 1,024 dimensional embeddings for any text.

To embed a single string:

```bash
llm embed -m azure-gpt-4o-embed -c 'this is text'
```
This will return a JSON array of 1,024 floating point numbers.

The [LLM documentation](https://llm.datasette.io/en/stable/embeddings/index.html) has more, including how to embed in bulk and store the results in a SQLite database.

See [LLM now provides tools for working with embeddings](https://simonwillison.net/2023/Sep/4/llm-embeddings/) and [Embeddings: What they are and why they matter](https://simonwillison.net/2023/Oct/23/embeddings/) for more about embeddings.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-azure-openai
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
pytest
```