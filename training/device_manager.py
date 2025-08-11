import torch
import gc
from contextlib import nullcontext
from torch.cuda.amp import GradScaler, autocast
from .config import Config


class DeviceManager:
    def __init__(self, config: Config):
        self.device = torch.device(config.device)
        self.use_amp = config.use_mixed_precision and config.device != 'cpu'
        self.scaler = GradScaler() if self.use_amp else None
    
    def to_device(self, tensor_or_model):
        """Move tensor or model to the configured device"""
        return tensor_or_model.to(self.device)
    
    def autocast_context(self):
        """Return appropriate autocast context for mixed precision"""
        return autocast() if self.use_amp else nullcontext()
    
    def backward_and_step(self, loss, optimizer):
        """Handle backward pass with optional mixed precision"""
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
    
    def cleanup_memory(self):
        """Clean up device memory"""
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / 1024**3
        elif self.device.type == 'mps':
            return torch.mps.current_allocated_memory() / 1024**3
        else:
            # Fallback for CPU - return a reasonable estimate
            return 1.0
    
    def get_device_info(self) -> dict:
        """Get device information"""
        info = {'device': str(self.device), 'mixed_precision': self.use_amp}
        
        if self.device.type == 'cuda':
            info['gpu_name'] = torch.cuda.get_device_name()
            info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return info