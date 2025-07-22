import os
from typing import Any, Dict, List, Tuple, Optional  


import numpy as np
import onnxruntime as ort
import torch  # NOQA
from cryptography.fernet import Fernet

from .tiling_image_loader import SA_Tiling_ImageLoader


class SA_SuperResolution:
    """
    A class to apply super-resolution on images using a specific model.
    """

    def __init__(
        self,
        models_dir: str,
        model_scale: int,
        tile_size: int = 128,
        gpu_id: int = 0,
        verbosity: bool = False,
    ) -> None:
        """
        Initializes the SA_SuperResolution class with the model path and scale.

        Parameters:
            models_dir (str): The path to the encrypted models dir.
            model_scale (int): The scale factor for the super-resolution process.
            tile_size (int): The tile size to pass to the image loader. Default is 128.
            gpu_id (int): Index of the GPU to use. CPU is -1. Default is 0.
            verbosity (bool): Verbosity of the model printing. Defaults is False.
        """
        self.scale: int = model_scale
        self.tile_size: int = tile_size
        self.encrypted_model_path: str = self._model_definition(models_dir)

        self.network, self.input_name = self._decrypt_model(gpu_id, verbosity)
        self.dataloader: SA_Tiling_ImageLoader = SA_Tiling_ImageLoader(self.tile_size)

    def _model_definition(self, models_dir: str) -> str:
        return os.path.join(models_dir, f"edsr_{self.scale}x.ven")

    def _decrypt_model(self, gpu_id: int, verbosity: bool = False, decryption_key: Optional[bytes] = None
        ) -> Tuple[ort.InferenceSession, str]:
 
        """
        Decodes an encrypted ONNX model and initializes an ONNX Runtime inference session.

        Args:
            gpu_id (int): Index of the GPU to use. CPU is -1
            verbosity (bool): Verbosity of the model printing. Defaults is False.
            decryption_key (bytes, optional): The key used to decrypt the model. If not provided, it uses a default key.

        Returns:
            Tuple[ort.InferenceSession, str]: A tuple containing the ONNX inference session and the model's input name.
        """
        with open(self.encrypted_model_path, "rb") as encrypted_file:
            encrypted_model = encrypted_file.read()

        if decryption_key is None:
            key = b"LtBDDJTE04l7Kef4PiYTa21RX4svq1vcGRbBkW_ZSwc="
        else:
            key = decryption_key

        fernet = Fernet(key)
        decrypted_model = fernet.decrypt(encrypted_model)

        providers: List[str | Tuple[str, Dict[str, Any]]] = [
            "CPUExecutionProvider"
        ]  # Always include CPU provider
        if gpu_id >= 0:
            providers.insert(
                0,
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": gpu_id,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
            )
        model = ort.InferenceSession(decrypted_model, providers=providers)
        input_name = model.get_inputs()[0].name
        model_name = os.path.basename(self.encrypted_model_path)
        if verbosity is True:
            if torch.cuda.is_available() and gpu_id >= 0:
                provider_list = model.get_providers()
                if "CUDAExecutionProvider" in provider_list:
                    print(f" -> Init {model_name} Neural Network on GPU")
                else:
                    print(
                        f" -> GPU available, but there was a problem initializing the model {model_name} on it. Initialization of the model in CPU"
                    )
            elif gpu_id < 0:
                print(f"Initializing {model_name} Neural Network on CPU")
            else:
                print(
                    f" -> CUDA not available. Initializing {model_name} Neural Network on CPU"
                )
        return model, input_name

    def _inference(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Performs inference on a single image tile using an ONNX model.

        Args:
            tile (torch.Tensor): The image tile on which to perform inference.

        Returns:
            torch.Tensor: The inference result as a PyTorch tensor.
        """
        input_tile = {self.input_name: tile.numpy()}
        output_tile = self.network.run(None, input_tile)
        output_tensor = torch.from_numpy(output_tile[0])
        return output_tensor

    def run(self, img_np: np.ndarray) -> np.ndarray:
        """
        Runs the super-resolution process on an input image.

        Parameters:
            img_np (np.ndarray): The numpy array representing the image in RGB format.

        Returns:
            np.ndarray: The super-resolved output image.
        """
        img_tiles, original_shape, padded_shape = self.dataloader.load_image(img_np)

        output_tiles = [self._inference(tile) for tile in img_tiles]

        output_img = self.dataloader.reconstruct_image_from_tiles_with_blending(
            output_tiles,
            padded_shape,
            self.scale,
        )

        output_img = output_img[
            :, : original_shape[0] * self.scale, : original_shape[1] * self.scale
        ]

        output_img = output_img.squeeze().cpu().numpy().transpose(1, 2, 0)
        out_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)

        return out_img
