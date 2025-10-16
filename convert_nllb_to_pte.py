import sys
import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1️⃣ Parse args and load model/tokenizer from Hugging Face
parser = argparse.ArgumentParser(description="Export HF seq2seq model to .pte for ExecuTorch")
parser.add_argument("model_name", nargs="?", default="facebook/m2m100_418M", help="Hugging Face model id or local path")
parser.add_argument("--quantize-dynamic", action="store_true", help="Apply dynamic int8 quantization to Linear layers before tracing (CPU-only)")
parser.add_argument("--output", default=None, help="Output .pte path (defaults to <model_id_last_token>.pte)")
args = parser.parse_args()

model_name = args.model_name
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
except Exception as e:
    msg = str(e)
    if "bitsandbytes" in msg or "load_in_4bit" in msg or "quantization" in msg:
        print(
            "This model appears to require bitsandbytes 4-bit quantization and cannot be TorchScript-exported on CPU/ExecuTorch.\n"
            "Use the base FP model instead (e.g., facebook/nllb-200-distilled-600M) or a CPU-friendly alternative (e.g., facebook/m2m100_418M).\n"
            "Model: " + model_name
        )
        sys.exit(1)
    raise
model.eval()
# Disable KV cache to keep outputs simple and stable for tracing
if getattr(model.config, "use_cache", None) is not None:
    model.config.use_cache = False

# Guard against quantized runtime models that are incompatible with TorchScript/ExecuTorch
if getattr(model, "is_loaded_in_4bit", False) or getattr(getattr(model, "quantization_config", None), "load_in_4bit", False):
    print(
        "Detected 4-bit quantized model (bitsandbytes). This is not supported for TorchScript/.pte export.\n"
        "Please use the base FP checkpoint (facebook/nllb-200-distilled-600M) or another CPU-friendly model."
    )
    sys.exit(1)

# 2️⃣ Prepare dummy input for tracing
# (NLLB is a seq2seq model, so we'll use a small example sentence)
sample_text = "Hello, how are you?"
inputs = tokenizer(sample_text, return_tensors="pt")

# For TorchScript tracing, we only need tensors
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 3️⃣ Trace the model for PyTorch Mobile (TorchScript scripting often fails for HF models)
print("Converting model to TorchScript via tracing...")

# Encoder-decoder models need decoder_input_ids during tracing
decoder_start_id = (
    model.config.decoder_start_token_id
    if getattr(model.config, "decoder_start_token_id", None) is not None
    else model.config.bos_token_id
)
decoder_input_ids = torch.full((1, 1), decoder_start_id, dtype=torch.long)

class EncoderDecoderWrapper(torch.nn.Module):
    def __init__(self, seq2seq_model: torch.nn.Module):
        super().__init__()
        self.seq2seq_model = seq2seq_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.seq2seq_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        return outputs.logits

wrapper = EncoderDecoderWrapper(model)

with torch.inference_mode():
    traced_model = torch.jit.trace(wrapper, (input_ids, attention_mask, decoder_input_ids), strict=False)

# Optional: apply dynamic quantization to reduce size (works for Linear layers on CPU)
if args.quantize_dynamic:
    try:
        print("Applying dynamic int8 quantization to Linear layers...")
        model.cpu()
        wrapper_cpu = EncoderDecoderWrapper(model).cpu()
        wrapper_cpu.eval()
        wrapper_cpu = torch.quantization.quantize_dynamic(wrapper_cpu, {torch.nn.Linear}, dtype=torch.qint8)
        with torch.inference_mode():
            traced_model = torch.jit.trace(wrapper_cpu, (input_ids.cpu(), attention_mask.cpu(), decoder_input_ids.cpu()), strict=False)
        print("Dynamic quantization applied.")
    except Exception as e:
        print(f"⚠️ Dynamic quantization failed, continuing without it: {e}")

# 4️⃣ Save as .pte (PyTorch Edge format)
default_name = model_name.split("/")[-1].replace(":", "_")
output_pte_path = args.output or f"{default_name}.pte"
traced_model._save_for_lite_interpreter(output_pte_path)
print(f"✅ Model exported to: {output_pte_path}")

# 5️⃣ Verify loading with Lite Interpreter (if supported in this PyTorch build)
print("Testing .pte model load...")
load_lite = getattr(torch.jit, "_load_for_lite_interpreter", None)
if load_lite is None:
    print("ℹ️ Current PyTorch build lacks _load_for_lite_interpreter; skipping load test.")
else:
    lite_model = load_lite(output_pte_path)
    lite_model.eval()
    print("✅ Successfully loaded .pte model for mobile use!")
