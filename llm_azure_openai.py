import click
import llm
from pydantic import Field
from openai import AzureOpenAI
from openai.types import Model
from typing import Optional
import json
import httpx
import os

DEFAULT_ALIASES = {
    "gpt-4o-128k-latest": "azure-gpt-4o",
    "gpt-4o-mini-128k-latest": "azure-gpt-4o-mini",
    "gpt-35-turbo-4k": "azure-gpt-35",
}

class ModelEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Model):
            return obj.__dict__ 
        return super().default(obj)

@llm.hookimpl
def register_models(register):
    # Currently not in use due to the Azure API for fetching deployed models not being available
    # for model_id in get_model_ids():
    #     alias = DEFAULT_ALIASES.get(model_id)
    #     aliases = [alias] if alias else []
    #     register(Azure_OpenAI(model_id), aliases=aliases)
    
    # Registering models manually for now
    for model_id in DEFAULT_ALIASES.keys():
        register(Azure_OpenAI(model_id), aliases=[DEFAULT_ALIASES.get(model_id)])
    
def list_azure_deployments():
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")

    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }

    response = httpx.get(f"{azure_endpoint}/openai/deployments", headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to retrieve deployments: {response.status_code} - {response.text}")

def refresh_models():
    click.echo("Not implemented. Please use the Azure API to fetch the list of models.")
    return []
    
    click.echo("Refreshing Azure OpenAI models")
    user_dir = llm.user_dir()
    azure_openai_models = user_dir / "azure_openai_models.json"
    
    api_key = llm.get_key("", "azure", "AZURE_OPENAI_API_KEY")
    azure_endpoint = llm.get_key("", "azure_url", "AZURE_OPENAI_ENDPOINT")
    if not api_key:
        raise click.ClickException(
            "You must set the 'azure' key or the AZURE_OPENAI_API_KEY environment variable."
        )
    client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-05-01-preview",
            azure_endpoint=azure_endpoint
        )
    models = client.models.list()

    chat_completion_models = [
        model for model in models if model.capabilities.get('chat_completion', False)
    ]

    print(f"Saving {len(chat_completion_models)} chat completion models to {azure_openai_models}")
    azure_openai_models.write_text(json.dumps(chat_completion_models, indent=2, cls=ModelEncoder))
    return chat_completion_models
    
def get_model_ids():
    user_dir = llm.user_dir()
    azure_openai_models = user_dir / "azure_openai_models.json"
    
    if azure_openai_models.exists():
        models = json.loads(azure_openai_models.read_text())
    elif llm.get_key("", "azure", "AZURE_OPENAI_API_KEY"):
        models = refresh_models()
    else:
        models = {list[Model]}

    return [model["id"] for model in models]

@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def azure_openai():
        "Commands relating to the llm-mistral plugin"
        
    @azure_openai.command()
    def models():
        "List available Azure OpenAI models"
        for model_id in get_model_ids():
            click.echo(model_id)

    @azure_openai.command()
    def refresh():       
        "Refresh the list of available Mistral models"
        before = set(get_model_ids())
        refresh_models()
        after = set(get_model_ids())
        added = after - before
        removed = before - after
        if added:
            click.echo(f"Added models: {', '.join(added)}", err=True)
        if removed:
            click.echo(f"Removed models: {', '.join(removed)}", err=True)
        if added or removed:
            click.echo("New list of models:", err=True)
            for model_id in get_model_ids():
                click.echo(model_id, err=True)
        else:
            click.echo("No changes", err=True)
            

class Azure_OpenAI(llm.Model):
    
    def __init__(self, model_id):
        self.model_id = model_id
    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description=(
                "Determines the sampling temperature. Higher values like 0.8 increase randomness, "
                "while lower values like 0.2 make the output more focused and deterministic."
            ),
            ge=0,
            le=1,
            default=0.5,
        )
        top_p: Optional[float] = Field(
            description=(
                "Nucleus sampling, where the model considers the tokens with top_p probability mass. "
                "For example, 0.1 means considering only the tokens in the top 10% probability mass."
            ),
            ge=0,
            le=1,
            default=0.5,
        )
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate in the completion.",
            ge=0,
            default=None,
        )
        frequency_penalty: Optional[float] = Field(
            description=(
                "The frequency penalty to reduce the likelihood of the model repeating the same response."
            ),
            ge=0,
            default=0,
        )
        presence_penalty: Optional[float] = Field(
            description=(
                "The presence penalty to reduce the likelihood of the model generating a response that is "
                "too similar to the context."
            ),
            ge=0,
            default=0,
        )
        deployment_name: Optional[str] = Field(
            description="The deployment name of the model",
            default=None,
        )
        
    def build_messages(self, prompt: llm.Prompt, conversation: llm.Conversation):
        if not conversation:
            messages = []
            if prompt.system:
                messages.append({
                    "role": "system",
                    "content": prompt.system
                })
            messages.append({
                "role": "user",
                "content": prompt.prompt
            })
            return messages
            
        messages = []
        for response in conversation.responses:
            messages.append({
                "role": "user",
                "content": response.prompt.prompt
            })
            
        messages.append({
            "role": "user",
            "content": prompt.prompt
        })
        return messages

    def execute(self, prompt, stream, response, conversation):
        api_key = llm.get_key("", "azure", "AZURE_OPENAI_API_KEY")
        azure_endpoint = llm.get_key("", "azure_url", "AZURE_OPENAI_ENDPOINT")
        
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-05-01-preview",
            azure_endpoint=azure_endpoint
        )
        
        deployment_name = prompt.options.deployment_name if prompt.options.deployment_name else self.model_id
        temperature = prompt.options.temperature
        top_p = prompt.options.top_p
        max_tokens = prompt.options.max_tokens
        frequency_penalty = prompt.options.frequency_penalty
        presence_penalty = prompt.options.presence_penalty
        
        response_ = client.chat.completions.create(
            model=deployment_name,
            messages=self.build_messages(prompt, conversation),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=None,
            stream=False
        )
        
        yield response_.choices[0].message.content
        response.response_json = response_.to_json()