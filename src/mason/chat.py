import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import typer
from rich.console import Console

console = Console()

def chat(
    model_path: str = typer.Argument(..., help="Path to the model (e.g., ./merged_model) or HF ID."),
    prompt: str = typer.Argument(..., help="Text prompt to send to the model."),
    system: str = typer.Option(None, help="Optional system prompt to set context (e.g., 'You are a pirate')."),
    device: str = typer.Option("auto", help="Device to use (e.g., 'cpu', 'cuda', 'auto')."),
    max_new_tokens: int = typer.Option(1024, help="Maximum number of tokens to generate."),
    use_template: bool = typer.Option(True, help="Use the tokenizer's chat template if available.")
):
    """
    Simple chat command to generate text from a model using its specific chat template.
    """
    console.print(f"[bold blue]Loading model from:[/bold blue] {model_path}")

    try:
        # Load Model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            return_dict=True,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        console.print(f"[bold yellow]Generating response...[/bold yellow]")

        # 1. Construct the Message History
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        # 2. Apply Chat Template (The magic part)
        # This converts [{"role": "user", "content": "hi"}] -> "[INST] hi [/INST]" (or whatever the model needs)
        if use_template and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                console.print("[dim]Applying chat template...[/dim]")
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True, # Tells model: "It's your turn to speak now"
                    return_tensors="pt"
                ).to(model.device)
            except Exception as template_error:
                console.print(f"[bold red]Template Error:[/bold red] {template_error}")
                console.print("[dim yellow]Falling back to raw prompt...[/dim yellow]")
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        else:
            # Fallback: No template found or requested, just raw text
            if use_template:
                console.print("[dim yellow]No chat template found in tokenizer, using raw prompt.[/dim yellow]")

            raw_text = prompt
            if system:
                raw_text = f"System: {system}\nUser: {prompt}\nAssistant:"

            input_ids = tokenizer(raw_text, return_tensors="pt").input_ids.to(model.device)

        # 3. Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7
            )

        # 4. Decode only the new tokens (response)
        # We slice the output to exclude the input prompt so we don't print the question again
        generated_ids = outputs[0][len(input_ids[0]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        console.print("\n[bold green]--- Model Output ---[/bold green]")
        console.print(response)
        console.print("[bold green]--------------------[/bold green]\n")

    except Exception as e:
        console.print(f"[bold red]Error during generation:[/bold red] {e}")
        raise typer.Exit(code=1)