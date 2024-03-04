import logging
import transformers
from transformers import Trainer
import inspect
from typing import Dict, Union, Any
import torch
import json
from torch import nn
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from image_utils import flattened_patches_to_image
import wandb
import numpy as np
import torch.distributed as dist
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.optimization import get_scheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import math
import os
import subprocess
from packaging import version
import accelerate
from transformers.trainer_pt_utils import find_batch_size, nested_concat, nested_numpify

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers.trainer_utils import seed_worker


logger = logging.getLogger(__name__)

def is_ge_version(v):
    return version.parse(transformers.__version__) >= version.parse(v) 

def _set_signature_columns_if_needed(self):
    if self._signature_columns is None:
        # Inspect model forward signature to keep only the arguments it accepts.
        signature = inspect.signature(self.model.forward)
        self._signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        self._signature_columns += list(set(["label", "label_ids", "tokens", "image", "font_size", "text", "patch_mask"] + self.label_names))


def compute_loss(self, model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    outputs = model(**inputs)
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    
    # Extra logs
    mae_loss_key = "pixel_loss" if hasattr(outputs, "pixel_loss") else "mae_loss"
    logits_key = "patch_logits" if hasattr(outputs, "patch_logits") else "logits"
    prefix = "eval_" if return_outputs else ""

    if not dist.is_initialized() or dist.get_rank() == 0: # This is an ugly way to log stuff and probably not thread-safe so only log it on rank == 0
        if not hasattr(self, "extra_logs"):
            self.extra_logs = {}
        if hasattr(outputs, mae_loss_key):
            self.extra_logs[prefix + mae_loss_key] = outputs[mae_loss_key].item()
            if self.args.log_eval_image_pred and return_outputs is True:
                images = [
                    flattened_patches_to_image(
                        outputs[logits_key][i].detach().cpu().to(torch.float32), 
                        height=self.args.height, 
                        width=self.args.width, 
                        patch_height=self.args.patch_height, 
                        patch_width=self.args.patch_width, 
                        image_mode=getattr(self.args, 'image_mode', 'RGB')
                    ) 
                    for i in range(len(outputs[logits_key]))
                ] # WARNING: I didn't set the size here
                self.extra_logs[prefix + "image_pred"] = [wandb.Image(image) for image in images]

                images = [
                    flattened_patches_to_image(
                        inputs["flattened_patches"][i, :, 2:].detach().cpu().to(torch.float32), 
                        height=self.args.height, 
                        width=self.args.width, 
                        patch_height=self.args.patch_height, 
                        patch_width=self.args.patch_width, 
                        image_mode=getattr(self.args, 'image_mode', 'RGB')
                    ) 
                    for i in range(len(inputs["flattened_patches"]))
                ] 
                self.extra_logs[prefix + "image_input"] = [wandb.Image(image) for image in images]

                if hasattr(outputs, "mask"):
                    images = [
                        flattened_patches_to_image(
                            outputs[logits_key][i].detach().cpu().to(torch.float32),
                            mask=outputs["mask"][i].detach().cpu().long(),
                            original_patches=inputs["flattened_patches"][i, :, 2:].detach().cpu().to(torch.float32),
                            height=self.args.height, width=self.args.width,
                            patch_height=self.args.patch_height, 
                            patch_width=self.args.patch_width, 
                            image_mode=getattr(self.args, 'image_mode', 'RGB')
                        ) 
                        for i in range(len(outputs[logits_key]))
                    ] # WARNING: I didn't set the size here
                    self.extra_logs[prefix + "image_pred_mask"] = [wandb.Image(image) for image in images]
        elif "flattened_patches" in inputs and self.args.log_eval_image_pred and return_outputs is True:
            images = [
                flattened_patches_to_image(
                    inputs["flattened_patches"][i, :, 2:].detach().cpu().to(torch.float32), 
                    height=self.args.height, 
                    width=self.args.width, 
                    patch_height=self.args.patch_height, 
                    patch_width=self.args.patch_width, 
                    image_mode=getattr(self.args, 'image_mode', 'RGB')
                ) 
                for i in range(len(inputs["flattened_patches"]))
            ] 
            self.extra_logs[prefix + "image_input"] = [wandb.Image(image) for image in images]

            if hasattr(outputs, "mask"):  # ViT MAE
                images = [
                    flattened_patches_to_image(
                        outputs[logits_key][i].detach().cpu().to(torch.float32), 
                        height=self.args.height, 
                        width=self.args.width, 
                        patch_height=self.args.patch_height, 
                        patch_width=self.args.patch_width, 
                        image_mode=getattr(self.args, 'image_mode', 'RGB')
                    ) 
                    for i in range(len(outputs[logits_key]))
                ] # WARNING: I didn't set the size here
                self.extra_logs[prefix + "image_pred"] = [wandb.Image(image) for image in images]

                images = [flattened_patches_to_image(
                        outputs[logits_key][i].detach().cpu().to(torch.float32),
                        mask=outputs["mask"][i].detach().cpu().long(),
                        original_patches=inputs["flattened_patches"][i, :, 2:].detach().cpu().to(torch.float32),
                        height=self.args.height, width=self.args.width,
                        patch_height=self.args.patch_height, 
                        patch_width=self.args.patch_width, 
                        image_mode=getattr(self.args, 'image_mode', 'RGB')
                    ) 
                    for i in range(len(outputs[logits_key]))] # WARNING: I didn't set the size here
                self.extra_logs[prefix + "image_pred_mask"] = [wandb.Image(image) for image in images]

        if hasattr(outputs, "text_loss"):
            self.extra_logs[prefix + "text_loss"] = outputs["text_loss"] if isinstance(outputs["text_loss"], float) else outputs["text_loss"].item()
        
        if hasattr(outputs, "dice_loss") and outputs.dice_loss is not None:
            self.extra_logs[prefix + "dice_loss"] = outputs["dice_loss"] if isinstance(outputs["dice_loss"], float) else outputs["dice_loss"].item()

    # Gather losses for logging
    if is_ge_version("4.34.1"):
        # accelerator gather only applies to >=4.34.1

        if not hasattr(self, "extra_logs"):
            self.extra_logs = {} 

        batch_size = find_batch_size(inputs)

        if hasattr(outputs, "text_loss"):
            text_losses = self.accelerator.gather_for_metrics(outputs["text_loss"].mean().detach().repeat(batch_size))
            self.extra_logs[prefix+"text_loss_aggr"] = text_losses if prefix+"text_loss_aggr" not in self.extra_logs else nested_concat(self.extra_logs[prefix+"text_loss_aggr"], text_losses)
        if hasattr(outputs, "dice_loss") and outputs.dice_loss is not None:
            dice_losses = self.accelerator.gather_for_metrics(outputs["dice_loss"].mean().detach().repeat(batch_size))
            self.extra_logs[prefix+"dice_loss_aggr"] = dice_losses if prefix+"dice_loss_aggr" not in self.extra_logs else nested_concat(self.extra_logs[prefix+"dice_loss_aggr"], dice_losses)
        if hasattr(outputs, mae_loss_key):
            mae_losses = self.accelerator.gather_for_metrics(outputs[mae_loss_key].mean().detach().repeat(batch_size))
            self.extra_logs[prefix+mae_loss_key+"_aggr"] = mae_losses if prefix+mae_loss_key+"_aggr" not in self.extra_logs else nested_concat(self.extra_logs[prefix+mae_loss_key+"_aggr"], mae_losses)

    return (loss, outputs) if return_outputs else loss

# New
def compute_loss_wrapper(self, model, inputs, return_outputs=False):
    saved_kwargs = {}
    if "true_labels" in inputs:
        saved_kwargs["true_labels"] = inputs.pop("true_labels")
    
    loss_and_outputs = compute_loss(self, model, inputs, return_outputs=return_outputs)
    if isinstance(loss_and_outputs, tuple) and len(loss_and_outputs) == 2:
        loss, outputs = loss_and_outputs
        if isinstance(outputs, tuple):
            for k in saved_kwargs:
                outputs = outputs + (saved_kwargs[k],)
        elif isinstance(outputs, dict):
            for k in saved_kwargs:
                outputs[k] = saved_kwargs[k]
        else:
            for k in saved_kwargs:
                setattr(outputs, k, saved_kwargs[k])
        return (loss, outputs)
    else:
        return loss_and_outputs

def log(self, logs: Dict[str, float]) -> None:
    """
    Log `logs` on the various objects watching training.

    Subclass and override this method to inject custom behavior.

    Args:
        logs (`Dict[str, float]`):
            The values to log.
    """
    if self.state.epoch is not None:
        logs["epoch"] = round(self.state.epoch, 2)
    
    logs["step"] = self.state.global_step
    
    if hasattr(self, "extra_logs"):
        for key in self.extra_logs:
            if "aggr" in key:
                h = nested_numpify(self.extra_logs[key])
                logs.update({key: h.mean().item()})
            else:
                logs.update({key: self.extra_logs[key]})

        self.extra_logs = {}

    self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    # Pop up the image type because they can't be saved
    pop_keys = []
    for key in logs:
        if "image_pred" in key or "image_input" in key:
            pop_keys.append(key)
    for key in pop_keys:
        logs.pop(key)

    output = {**logs}
    self.state.log_history.append(output)



import signal
from subprocess import call
class SIGUSR1Callback(transformers.TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self.signal_received = False
        signal.signal(signal.SIGUSR1, self.handle_signal)
        # signal.signal(signal.SIGINT, self.handle_signal)
        logger.warn("Handler registered")

    def handle_signal(self, signum, frame):
        self.signal_received = True
        logger.warn("Signal received")

    def on_step_end(self, args, state, control, **kwargs):
        if self.signal_received:
            control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.signal_received:
            exit(0)



def _pad_tensors_to_max_len(self, tensor, max_length):
    if self.model.config.pad_token_id is not None:
        pad_token_id = self.model.config.pad_token_id
    else:
        raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

    padded_tensor = pad_token_id * torch.ones(
        (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, : tensor.shape[-1]] = tensor
    return padded_tensor
    

def prediction_step_seq2seq(
    self,
    model: nn.Module,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    prediction_loss_only: bool,
    ignore_keys: Optional[List[str]] = None,
) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Copied from HF's seq2seq_trainer.py

    Perform an evaluation step on `model` using `inputs`.

    Subclass and override to inject custom behavior.

    Args:
        model (`nn.Module`):
            The model to evaluate.
        inputs (`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument `labels`. Check your model's documentation for all accepted arguments.
        prediction_loss_only (`bool`):
            Whether or not to return the loss only.
        gen_kwargs:
            Additional `generate` specific kwargs.

    Return:
        Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
        labels (each being optional).
    """

    has_labels = "labels" in inputs
    inputs = self._prepare_inputs(inputs)

    # XXX: adapt synced_gpus for fairscale as well
    # Priority (handled in generate):
    # gen_kwargs > model.generation_config > default GenerationConfig()
    gen_kwargs = self._gen_kwargs

    # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
    # (otherwise, it would continue generating from the padded `decoder_input_ids`)
    if (
        "labels" in inputs
        and "decoder_input_ids" in inputs
        and inputs["labels"].shape == inputs["decoder_input_ids"].shape
    ):
        inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
    
    # New
    true_labels = inputs.pop("true_labels", None)
    generated_tokens = self.model.generate(**inputs, **gen_kwargs)

    # in case the batch is shorter than max length, the output should be padded
    # if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
    #     generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
    if gen_kwargs["max_new_tokens"] is not None and generated_tokens.shape[-1] < gen_kwargs["max_new_tokens"] + 1:
        generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

    with torch.no_grad():
        if has_labels:
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            if self.label_smoother is not None:
                loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
            else:
                loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
        else:
            loss = None

    if self.args.prediction_loss_only:
        return loss, None, None

    if has_labels:
        labels = inputs["labels"]
        # if labels.shape[-1] < gen_config.max_length:
        #     labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
        # if gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
        #     labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        if gen_kwargs["max_new_tokens"] is not None and labels.shape[-1] < gen_kwargs["max_new_tokens"] + 1:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"] + 1)
    else:
        labels = None

    # New
    if true_labels is not None:
        labels = true_labels

    return loss, generated_tokens, labels


def get_cosine_schedule_to_min_lr_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    max_lr: float,
    min_lr: float = 1e-5,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to a minimum learning rate, after a warmup period during which it increases linearly
    between 0 and the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        max_lr (`float`):
            The maximum learning rate after warming up, right before decaying
        min_lr (`float`):
            The minimum learning rate at the end of training
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to the min
            value following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return (
            max(
                min_lr,
                min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
            )
            / max_lr  # Scale down by max_lr because LambdaLR multiplies back by max_lr
        )

    logger.info("***** Creating cosine scheduler to min_lr with warmup *****")
    logger.info(f"\t{num_warmup_steps = }")
    logger.info(f"\t{num_training_steps = }")
    logger.info(f"\t{max_lr = }")
    logger.info(f"\t{min_lr = }")
    logger.info(f"\t{num_cycles = }")
    logger.info(f"\t{last_epoch = }")

    return LambdaLR(optimizer, lr_lambda, last_epoch)



def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    passed as an argument.

    Args:
        num_training_steps (int): The number of training steps to do.
    """
    if self.lr_scheduler is None:
        if self.args.lr_scheduler_type == "cosine" and self.args.cosine_w_min:
            self.lr_scheduler = get_cosine_schedule_to_min_lr_with_warmup(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                max_lr=self.args.learning_rate,
                min_lr=self.args.min_learning_rate
            )
        else:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
    return self.lr_scheduler


def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (`nn.Module`):
            The model to train.
        inputs (`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument `labels`. Check your model's documentation for all accepted arguments.

    Return:
        `torch.Tensor`: The tensor with training loss on this batch.
    """
    model.train()
    inputs = self._prepare_inputs(inputs)

    with self.compute_loss_context_manager():
        loss = self.compute_loss(model, inputs)

    if self.args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training

    if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
        # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
        loss = loss / self.args.gradient_accumulation_steps

    if self.do_grad_scaling:
        self.scaler.scale(loss).backward()
    elif self.use_apex:
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
    elif self.deepspeed:
        # loss gets scaled under gradient_accumulation_steps in deepspeed
        loss = self.deepspeed.backward(loss)
    else:
        if is_ge_version("4.34.1"):
            self.accelerator.backward(loss)
        else:
            loss.backward()

    if getattr(self.args, "log_grad_norm", False):
        if not hasattr(self, "extra_logs"):
            self.extra_logs = {}
        # Go through all the parameters and log the gradient norm
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.extra_logs[f"grad_norm_{name}"] = torch.norm(param.grad.detach()).item()
    
    if getattr(self.args, "log_train_input", False) and "flattened_patches" in inputs:
        if not hasattr(self, "extra_logs"):
            self.extra_logs = {}
        images = [
            flattened_patches_to_image(
                inputs["flattened_patches"][i, :, 2:].detach().cpu().to(torch.float32), 
                height=self.args.height, 
                width=self.args.width, 
                patch_height=self.args.patch_height, 
                patch_width=self.args.patch_width, 
                image_mode=getattr(self.args, 'image_mode', 'RGB')
            ) for i in range(len(inputs["flattened_patches"]))
        ] 
        # We save those images on the disk, in a folder that is named by the step
        # First create the folder (naming: step_rank)
        # os.makedirs(f"image_logs/{self.state.global_step}_{dist.get_rank()}", exist_ok=True)
        # # Save images
        # for i in range(len(images)):
        #     images[i].save(f"image_logs/{self.state.global_step}_{dist.get_rank()}/{i}.png")            
        self.extra_logs["train_image_input"] = [wandb.Image(image) for image in images]

    return loss.detach()


from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

def _save_checkpoint(self, model, trial, metrics=None):
    # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
    # want to save except FullyShardedDDP.
    # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

    # Save model checkpoint
    checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    if self.hp_search_backend is None and trial is None:
        self.store_flos()

    run_dir = self._get_output_dir(trial=trial)
    output_dir = os.path.join(run_dir, checkpoint_folder)

    self._original_save_checkpoint(model, trial, metrics=metrics)


def get_train_dataloader_for_streaming(self) -> DataLoader:
    """
    Because streaming handles the distributed data parallel by itself, we don't need special data loader.
    The plainest data loader is enough.
    """
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = self.train_dataset
    data_collator = self.data_collator
    data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers, # Streaming dataset is probably not multi-thread safe
        "pin_memory": self.args.dataloader_pin_memory,
    }

    # Streaming is iterable
    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker

    # Instead of use accelerate to prepare the dataloader, we just return a plain dataloader
    return DataLoader(train_dataset, **dataloader_params)


def get_eval_dataloader_for_streaming(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
    """
    Because streaming handles the distributed data parallel by itself, we don't need special data loader.
    The plainest data loader is enough.
    """
    if eval_dataset is None and self.eval_dataset is None:
        raise ValueError("Trainer: evaluation requires an eval_dataset.")
    eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
    data_collator = self.data_collator
    data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

    dataloader_params = {
        "batch_size": self.args.eval_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers, # Streaming dataset is probably not multi-thread safe
        "pin_memory": self.args.dataloader_pin_memory,
    }

    # Streaming is iterable
    if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)  
        dataloader_params["drop_last"] = self.args.dataloader_drop_last

    # Instead of use accelerate to prepare the dataloader, we just return a plain dataloader
    return DataLoader(eval_dataset, **dataloader_params) 


def trainer_addon(trainer, seq2seq=False, streaming_dataset=False):
    trainer._set_signature_columns_if_needed = _set_signature_columns_if_needed.__get__(trainer, Trainer)
    # New
    trainer.compute_loss = compute_loss_wrapper.__get__(trainer, Trainer)
    trainer.log = log.__get__(trainer, Trainer)
    trainer.create_scheduler = create_scheduler.__get__(trainer, Trainer)
    trainer.training_step = training_step.__get__(trainer, Trainer)
    trainer._original_save_checkpoint = trainer._save_checkpoint
    trainer._save_checkpoint = _save_checkpoint.__get__(trainer, Trainer)

    if streaming_dataset:
        trainer.get_train_dataloader = get_train_dataloader_for_streaming.__get__(trainer, Trainer)
        trainer.get_eval_dataloader = get_eval_dataloader_for_streaming.__get__(trainer, Trainer)

    trainer.add_callback(SIGUSR1Callback())
    if seq2seq:
        trainer.prediction_step = prediction_step_seq2seq.__get__(trainer, Trainer)
        trainer._pad_tensors_to_max_len = _pad_tensors_to_max_len.__get__(trainer, Trainer)
    
    return trainer
