"""CoLA model support for lm-evaluation-harness.

CoLA is a low-rank training method that uses a LLaMA-compatible architecture.
This module provides an lm-eval model class for evaluating CoLA checkpoints.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import transformers

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from training.CoLA.cola import ColaConfig, ColaForCausalLM # Import CoLA training code to ensure model classes are registered

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

eval_logger = logging.getLogger(__name__)


@register_model("cola")
@register_model("cola")
class CoLALM(HFLM):
    def __init__(
        self, 
        pretrained: str, 
        device: str = "cuda", 
        **kwargs: Any
    ):
        # 1. Initialize the HFLM parent WITHOUT the model argument.
        # We pass 'pretrained' so it sets up the tokenizer correctly.
        super().__init__(
            pretrained=pretrained, 
            device=device, 
            **kwargs
        )

        # 2. Now, manually load your CoLA model.
        # This happens AFTER super().__init__ so we don't cause argument conflicts.
        eval_logger.info(f"Loading CoLA config from {pretrained}")
        config = ColaConfig.from_pretrained(pretrained)
        
        eval_logger.info(f"Loading CoLA weights from {pretrained}")
        # Use from_pretrained to ensure weights are actually loaded into the architecture
        self._model = ColaForCausalLM.from_pretrained(
            pretrained,
            config=config
        ).to(self.device)

        print(self._model)
        
        # 3. Finalize
        self._model.eval()
        torch.set_grad_enabled(False)