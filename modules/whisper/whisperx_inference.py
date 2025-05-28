import os
from typing import Union, BinaryIO, Tuple, List, Callable
import gradio as gr
import numpy as np

from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.whisper.data_classes import Segment

# You need to install whisperx: pip install whisperx
import whisperx

class WhisperXInference(BaseTranscriptionPipeline):
    def __init__(self, model_dir, diarization_model_dir, uvr_model_dir, output_dir):
        super().__init__(model_dir, diarization_model_dir, uvr_model_dir, output_dir)
        self.device = self.get_device()
        self.current_model_size = "small"  # default, can be changed
        self.current_compute_type = "int8"    # default, can be changed
        self.model = whisperx.load_model(self.current_model_size, device=self.device, compute_type=self.current_compute_type)

    def update_model(self, model_size: str, compute_type: str, progress: gr.Progress = gr.Progress()):
        """
        Update the WhisperX model with a new model size or compute type.
        """
        progress(0, desc="Initializing WhisperX Model...")
        self.current_model_size = model_size
        self.current_compute_type = compute_type
        self.model = whisperx.load_model(model_size, device=self.device, compute_type=compute_type)
        progress(1, desc="WhisperX Model Loaded.")

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            progress: gr.Progress = gr.Progress(),
            file_format: str = "SRT",
            add_timestamp: bool = True,
            progress_callback: Callable = None,
            *pipeline_params,
            ) -> Tuple[List[Segment], float]:
        import time
        start_time = time.time()
        # WhisperX handles VAD and diarization internally
        audio = whisperx.load_audio(audio)
        result = self.model.transcribe(audio)
        segments = []
        for seg in result["segments"]:
            segments.append(Segment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                speaker=seg.get("speaker", None)
            ))
        elapsed = time.time() - start_time
        return segments, elapsed

    def transcribe(self, *args, **kwargs):
        # Not used, as run() does everything for WhisperX
        pass