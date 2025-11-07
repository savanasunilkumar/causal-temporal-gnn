"""Model quantization and ONNX export for deployment optimization."""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Optional, Dict, Any
import logging
import os


class ModelQuantizer:
    """
    Quantize PyTorch models for faster inference and smaller size.
    """

    def __init__(self, model: nn.Module, device='cpu'):
        """
        Initialize quantizer.

        Args:
            model: Model to quantize
            device: Device for quantization
        """
        self.model = model.to(device)
        self.device = device
        self.quantized_model = None

        self.logger = logging.getLogger("ModelQuantizer")

    def quantize_dynamic(self, dtype=torch.qint8):
        """
        Apply dynamic quantization (good for LSTMs, Linear layers).

        Args:
            dtype: Quantization data type

        Returns:
            Quantized model
        """
        self.logger.info("Applying dynamic quantization...")

        # Specify layers to quantize
        self.quantized_model = quant.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.GRU},  # Layers to quantize
            dtype=dtype
        )

        self.logger.info("Dynamic quantization complete")
        return self.quantized_model

    def quantize_static(self, calibration_dataloader):
        """
        Apply static quantization (requires calibration).

        Args:
            calibration_dataloader: DataLoader for calibration

        Returns:
            Quantized model
        """
        self.logger.info("Applying static quantization...")

        # Prepare model for quantization
        self.model.eval()
        self.model.qconfig = quant.get_default_qconfig('fbgemm')

        # Fuse modules if possible
        model_fused = self._fuse_modules()

        # Prepare for quantization
        model_prepared = quant.prepare(model_fused)

        # Calibration
        self.logger.info("Calibrating...")
        with torch.no_grad():
            for batch in calibration_dataloader:
                model_prepared(*batch)

        # Convert to quantized model
        self.quantized_model = quant.convert(model_prepared)

        self.logger.info("Static quantization complete")
        return self.quantized_model

    def quantize_qat(self, train_dataloader, optimizer, num_epochs=5):
        """
        Apply Quantization-Aware Training (best quality).

        Args:
            train_dataloader: Training data
            optimizer: Optimizer
            num_epochs: Number of training epochs

        Returns:
            Quantized model
        """
        self.logger.info("Applying quantization-aware training...")

        # Prepare model
        self.model.train()
        self.model.qconfig = quant.get_default_qat_qconfig('fbgemm')

        # Fuse and prepare
        model_fused = self._fuse_modules()
        model_prepared = quant.prepare_qat(model_fused)

        # Train with quantization
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                optimizer.zero_grad()
                loss = self._compute_loss(model_prepared, batch)
                loss.backward()
                optimizer.step()

            self.logger.info(f"QAT Epoch {epoch + 1}/{num_epochs}")

        # Convert to quantized model
        model_prepared.eval()
        self.quantized_model = quant.convert(model_prepared)

        self.logger.info("Quantization-aware training complete")
        return self.quantized_model

    def _fuse_modules(self):
        """Fuse consecutive operations for better quantization."""
        # Fuse Conv+BN+ReLU, Conv+ReLU, Linear+ReLU, etc.
        # This is model-specific and should be customized
        return self.model

    def _compute_loss(self, model, batch):
        """Compute loss for QAT training."""
        # Placeholder - should be customized for specific model
        outputs = model(*batch['inputs'])
        loss = nn.functional.cross_entropy(outputs, batch['labels'])
        return loss

    def compare_performance(self, test_data):
        """
        Compare original vs quantized model performance.

        Args:
            test_data: Test data for comparison

        Returns:
            Comparison results
        """
        import time

        results = {}

        # Original model
        self.model.eval()
        start = time.time()
        with torch.no_grad():
            _ = self.model(*test_data)
        original_time = time.time() - start

        # Quantized model
        if self.quantized_model is not None:
            self.quantized_model.eval()
            start = time.time()
            with torch.no_grad():
                _ = self.quantized_model(*test_data)
            quantized_time = time.time() - start

            speedup = original_time / quantized_time
        else:
            quantized_time = None
            speedup = None

        # Model sizes
        original_size = self._get_model_size(self.model)
        quantized_size = self._get_model_size(self.quantized_model) if self.quantized_model else None

        results = {
            'original_time': original_time,
            'quantized_time': quantized_time,
            'speedup': speedup,
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': original_size / quantized_size if quantized_size else None
        }

        return results

    def _get_model_size(self, model):
        """Get model size in MB."""
        if model is None:
            return None

        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024

        return size_mb

    def save_quantized(self, filepath: str):
        """Save quantized model."""
        if self.quantized_model is None:
            raise ValueError("No quantized model available")

        torch.save(self.quantized_model.state_dict(), filepath)
        self.logger.info(f"Quantized model saved to {filepath}")


class ONNXExporter:
    """
    Export PyTorch models to ONNX format for cross-platform deployment.
    """

    def __init__(self, model: nn.Module, device='cpu'):
        """
        Initialize ONNX exporter.

        Args:
            model: Model to export
            device: Device
        """
        self.model = model.to(device).eval()
        self.device = device
        self.logger = logging.getLogger("ONNXExporter")

    def export(self, dummy_input, output_path: str,
               input_names: Optional[list] = None,
               output_names: Optional[list] = None,
               dynamic_axes: Optional[Dict] = None,
               opset_version: int = 11):
        """
        Export model to ONNX format.

        Args:
            dummy_input: Example input for tracing
            output_path: Path to save ONNX model
            input_names: Names of input tensors
            output_names: Names of output tensors
            dynamic_axes: Dynamic axes specification
            opset_version: ONNX opset version

        Returns:
            Path to exported model
        """
        self.logger.info(f"Exporting model to ONNX: {output_path}")

        # Default names
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']

        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )

        self.logger.info(f"Model exported to {output_path}")

        # Verify export
        self._verify_onnx(output_path, dummy_input)

        return output_path

    def _verify_onnx(self, onnx_path: str, dummy_input):
        """Verify exported ONNX model."""
        try:
            import onnx
            import onnxruntime as ort

            # Check model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            self.logger.info("ONNX model is valid")

            # Test inference
            ort_session = ort.InferenceSession(onnx_path)

            # Prepare input
            if isinstance(dummy_input, torch.Tensor):
                ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            elif isinstance(dummy_input, tuple):
                ort_inputs = {
                    inp.name: val.cpu().numpy()
                    for inp, val in zip(ort_session.get_inputs(), dummy_input)
                }
            else:
                ort_inputs = dummy_input

            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)

            # Compare with PyTorch
            with torch.no_grad():
                torch_output = self.model(dummy_input)

            # Check similarity
            if isinstance(torch_output, torch.Tensor):
                torch_output = torch_output.cpu().numpy()
                diff = np.abs(torch_output - ort_outputs[0]).max()
                self.logger.info(f"Max difference between PyTorch and ONNX: {diff}")

                if diff < 1e-4:
                    self.logger.info("✓ ONNX export verified successfully")
                else:
                    self.logger.warning(f"⚠ Large difference detected: {diff}")

        except ImportError:
            self.logger.warning("onnx/onnxruntime not installed, skipping verification")
        except Exception as e:
            self.logger.error(f"Error verifying ONNX model: {str(e)}")

    def optimize_onnx(self, onnx_path: str, optimized_path: str):
        """
        Optimize ONNX model for inference.

        Args:
            onnx_path: Input ONNX model path
            optimized_path: Output optimized model path
        """
        try:
            from onnxruntime.transformers import optimizer

            # Optimize
            optimized_model = optimizer.optimize_model(
                onnx_path,
                model_type='bert',  # or 'gpt2', customize as needed
                num_heads=8,
                hidden_size=512
            )

            optimized_model.save_model_to_file(optimized_path)
            self.logger.info(f"Optimized model saved to {optimized_path}")

        except ImportError:
            self.logger.warning("onnxruntime.transformers not available for optimization")


class TorchScriptExporter:
    """
    Export models using TorchScript for PyTorch-based deployment.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize TorchScript exporter.

        Args:
            model: Model to export
        """
        self.model = model.eval()
        self.logger = logging.getLogger("TorchScriptExporter")

    def export_trace(self, example_inputs, output_path: str):
        """
        Export using torch.jit.trace.

        Args:
            example_inputs: Example inputs for tracing
            output_path: Output path

        Returns:
            Traced model
        """
        self.logger.info("Tracing model with TorchScript...")

        traced_model = torch.jit.trace(self.model, example_inputs)

        # Save
        traced_model.save(output_path)
        self.logger.info(f"Traced model saved to {output_path}")

        return traced_model

    def export_script(self, output_path: str):
        """
        Export using torch.jit.script.

        Args:
            output_path: Output path

        Returns:
            Scripted model
        """
        self.logger.info("Scripting model with TorchScript...")

        scripted_model = torch.jit.script(self.model)

        # Save
        scripted_model.save(output_path)
        self.logger.info(f"Scripted model saved to {output_path}")

        return scripted_model


def export_for_production(model: nn.Module, export_dir: str,
                          example_input, quantize=True, export_onnx=True,
                          export_torchscript=True):
    """
    Export model in multiple formats for production deployment.

    Args:
        model: Model to export
        export_dir: Directory to save exports
        example_input: Example input for tracing
        quantize: Whether to quantize model
        export_onnx: Whether to export ONNX
        export_torchscript: Whether to export TorchScript

    Returns:
        Dictionary of export paths
    """
    os.makedirs(export_dir, exist_ok=True)
    exports = {}

    # Original PyTorch model
    torch_path = os.path.join(export_dir, 'model.pt')
    torch.save(model.state_dict(), torch_path)
    exports['pytorch'] = torch_path

    # Quantized model
    if quantize:
        quantizer = ModelQuantizer(model)
        quantized_model = quantizer.quantize_dynamic()

        quant_path = os.path.join(export_dir, 'model_quantized.pt')
        quantizer.save_quantized(quant_path)
        exports['quantized'] = quant_path

    # ONNX export
    if export_onnx:
        onnx_exporter = ONNXExporter(model)
        onnx_path = os.path.join(export_dir, 'model.onnx')
        onnx_exporter.export(example_input, onnx_path)
        exports['onnx'] = onnx_path

    # TorchScript export
    if export_torchscript:
        ts_exporter = TorchScriptExporter(model)
        ts_path = os.path.join(export_dir, 'model_torchscript.pt')
        ts_exporter.export_trace(example_input, ts_path)
        exports['torchscript'] = ts_path

    logging.info(f"All exports complete. Saved to {export_dir}")
    logging.info(f"Export formats: {list(exports.keys())}")

    return exports
