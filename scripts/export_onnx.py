"""
Export the trained PyTorch model to ONNX format for browser inference.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.model import load_model, SequenceNet

def export_to_onnx():
    """Export the trained model to ONNX format."""
    
    models_dir = Path(__file__).parent.parent / "models"
    checkpoint_path = models_dir / "latest.pt"
    output_path = Path(__file__).parent.parent / "web" / "static" / "model.onnx"
    
    if not checkpoint_path.exists():
        print(f"Error: No model found at {checkpoint_path}")
        return False
    
    print(f"Loading model from {checkpoint_path}...")
    model, device, _ = load_model(str(checkpoint_path), device='cpu', compile_model=False)
    model.eval()
    
    # Create dummy input matching the expected shape: (batch, 8, 10, 10)
    dummy_input = torch.randn(1, 8, 10, 10, device='cpu')
    
    print(f"Exporting to ONNX at {output_path}...")
    
    # Export to ONNX using legacy API (avoids onnxscript dependency)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        },
        dynamo=False  # Use legacy exporter
    )
    
    print(f"✓ Model exported to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Verify the export structure
    print("\nVerifying ONNX export...")
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid!")
    except ImportError:
        print("(onnx package not available for validation, but export complete)")
    
    return True
    
if __name__ == "__main__":
    export_to_onnx()
