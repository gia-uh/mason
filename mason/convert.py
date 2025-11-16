import typer
from rich.console import Console
from llama_cpp import llama_model_gguf_from_folder, LLAMA_FTYPE
from pathlib import Path
from typing_extensions import Annotated

console = Console()

# Create a simple, user-friendly map of quantization names to the
# integer ftype enums required by llama_cpp.
# e.g., "Q4_K_M" -> LLAMA_FTYPE.LLAMA_FTYPE_Q4_K_M
FTYPE_MAP = {
    name.replace("LLAMA_FTYPE_", ""): ftype.value
    for name, ftype in LLAMA_FTYPE.__members__.items()
    if name.startswith("LLAMA_FTYPE_Q") or name in ("LLAMA_FTYPE_F16", "LLAMA_FTYPE_F32")
}
# A sorted list of the keys for the help text
VALID_QUANTS = sorted(FTYPE_MAP.keys(), key=lambda k: (k[0], int(k[1]) if k[1].isdigit() else 99, k))


def convert_to_gguf(
    model_dir: Annotated[Path, typer.Argument(
        help="Directory containing the Hugging Face model (e.g., ./merged_model)",
        exists=True,
        file_okay=False,
        resolve_path=True
    )],
    output_path: Annotated[Path, typer.Argument(
        help="Path to save the output GGUF file (e.g., ./my_model-q4_k_m.gguf)",
        resolve_path=True
    )]
):
    """
    Convert a Hugging Face model to GGUF format.
    """
    raise NotImplemented