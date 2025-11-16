import os
import typer
from rich.console import Console

console = Console()

def merge(
    base_model: str = typer.Option(..., help="Path or HuggingFace ID of the base model."),
    lora_adapter: str = typer.Option(..., help="Path or HuggingFace ID of the LoRA adapter."),
    output_dir: str = typer.Option("./merged_model", help="Directory to save the merged model."),
    device: str = typer.Option("auto", help="Device to use for loading (e.g., 'cpu', 'cuda', 'auto').")
):
    """
    Merge a LoRA adapter into a base model.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    console.print(f"[bold blue]Loading base model:[/bold blue] {base_model}")

    try:
        # Load Base Model
        # torch_dtype=torch.bfloat16 is generally recommended for newer models (Llama 2/3, Mistral)
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            return_dict=True,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        console.print(f"[bold blue]Loading LoRA adapter:[/bold blue] {lora_adapter}")

        # Load Adapter
        model = PeftModel.from_pretrained(base_model_obj, lora_adapter)

        console.print("[bold yellow]Merging weights...[/bold yellow]")

        # Merge and Unload
        merged = model.merge_and_unload()

        console.print(f"[bold green]Saving merged model to:[/bold green] {output_dir}")

        # Save
        merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        console.print("[bold green]Success! Merge complete.[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error during merge:[/bold red] {e}")
        raise typer.Exit(code=1)