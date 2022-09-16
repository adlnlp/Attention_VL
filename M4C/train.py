import argparse
import logging
import os
import random
from io import open
from pprint import pprint
import json

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

from evaluator import Evaluator
from sam.sa_m4c import SAM4C, BertConfig
from sam.task_utils import (clip_gradients, forward_model,
                            get_optim_scheduler, load_datasets)
from tools.registry import registry

# new model variants
from sam.sa_m4c_attn import SAM4CAttn
from sam.sa_m4c_attn_qwk import SAM4CAttnQWK
from sam.sa_m4c_attn_kwq import SAM4CAttnKWQ
from sam.sa_m4c_attn_biased_qwk import SAM4CAttnBiasedQWK
from sam.sa_m4c_attn_biased_kwq import SAM4CAttnBiasedKWQ
from sam.sa_m4c_attn_act_qwk import SAM4CAttnActQWK
from sam.datasets.processors import SymbolDict
import pickle

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# TODO: add model Map for constructing different models
ModelMap = {
    "sam4c": SAM4C,
    "sam4c-attn": SAM4CAttn,
    "sam4c-attn-qwk": SAM4CAttnQWK,
    "sam4c-attn-kwq": SAM4CAttnKWQ,
    "sam4c-attn-biased-qwk": SAM4CAttnBiasedQWK,
    "sam4c-attn-biased-kwq": SAM4CAttnBiasedKWQ,
    "sam4c-attn-act-qwk": SAM4CAttnActQWK,
}


def get_config():
    # load command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_train_epochs",
        default=100,
        type=int,
        help="Total training epochs",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument("--config", required=True, type=str, help="Experiment configuration file")

    parser.add_argument(
        "--tag", type=str, help="Experiment folder name", default="debug"
    )

    parser.add_argument(
        "--attn_type", type=str, help="type of attention compatibility functions for transformer layers", default="scaled-dot"
    )

    parser.add_argument("--pretrained_eval", default="", help="Path of pre-trained checkpoint")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    # Todo: Move below code to another function
    # Reproducibility seeds
    seed = task_cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("-" * 20 + "Command Line Config: " + "-" * 20)
    print(pprint(vars(args)))
    logger.info("-" * 20 + "Task File Config: " + "-" * 20)
    print(pprint(task_cfg))

    # Build save path
    save_path = os.path.join(task_cfg["output_dir"], args.tag)
    if not os.path.exists(save_path) and args.pretrained_eval == "":
        os.makedirs(save_path)

    # Dump all configs
    with open(os.path.join(save_path, "command.txt"), "w") as f:
        print(f"Command Line: \n {str(vars(args))} \n \n", file=f)
        print(f"Config File: \n {str(vars(task_cfg))} \n \n", file=f)

    # Add all configs to registry
    registry.update(vars(args))
    registry.update(task_cfg)

    return task_cfg, args, save_path


def main():
    task_cfg, args, save_path = get_config()
    checkpoint_path = os.path.join(save_path, "best_model.tar")
    base_lr = task_cfg["lr"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"Device: {device}, Numer of GPUs: {n_gpu}")

    # Original data loader - needs too much ram
    # dataloaders = load_datasets(task_cfg, ["train", "val", "test"])
    
    # temp loader
    if args.pretrained_eval != "":
        dataloaders = load_datasets(task_cfg, ["val", "test"])
    else:
        dataloaders = load_datasets(task_cfg, ["train", "val"])

    mmt_config = BertConfig.from_dict(task_cfg["SA-M4C"])
    text_bert_config = BertConfig.from_dict(task_cfg["TextBERT"])
    
    model = ModelMap[task_cfg["model_variant"]](mmt_config, text_bert_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Training Parameters: {trainable_params}")
    optimizer_grouped_parameters = model.get_optimizer_parameters(base_lr)
    print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    optimizer, warmup_scheduler = get_optim_scheduler(
        task_cfg, optimizer_grouped_parameters, base_lr
    )
    start_iter_id, global_step, start_epoch = 0, 0, 0
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # When running only evaluation
    if args.pretrained_eval != "":
        logger.info(
            f"Dumping Evaluation results at: {os.path.dirname(args.pretrained_eval)}"
        )
        
        # Temp return for evaluation result
        return args.pretrained_eval, model, dataloaders, task_cfg, device
        
        # original return
        # return args.pretrained_eval, model, dataloaders

    # This validation score is used for model-saving.
    best_val_step, best_val_score = -1, -1
    loss_values, score_values = [], []
    median_num_iter = len(dataloaders["train"])

    # Train loop
    model.train()
    for epoch_id in tqdm(range(start_epoch, args.num_train_epochs), desc="Epoch"):
        for step in tqdm(range(median_num_iter), desc="Iters"):
            assert model.training
            iter_id = start_iter_id + step + (epoch_id * median_num_iter)

            loss, score, _, _ = forward_model(
                task_cfg, device, model, dataloaders, "train"
            )

            # Compute gradients
            loss.backward()
            clip_gradients(model, task_cfg["max_grad_norm"])
            # if step == 194:
            #     for name, param in model.named_parameters():
            #         if param.grad == 0:
            #             print(name)
            # Apply and reset gradients
            optimizer.step()
            warmup_scheduler.step()
            model.zero_grad()

            # Increment loggers
            global_step += 1
            loss_values.append(loss)
            score_values.append(score)

            # Handle logging
            if step % 20 == 0 and step != 0:
                loss_avg, score_avg = float(sum(loss_values) / len(loss_values)), float(
                    sum(score_values) / len(score_values)
                )
                loss_values, score_values = [], []
                log_str = f"Epoch: {epoch_id}: Iter: {iter_id};  loss = {loss_avg}; accuracy  = {score_avg}"
                if step % 100 == 0:
                    log_str += f"\n lr rates = {[float(grp['lr']) for grp in optimizer.param_groups]}"
                logger.info(log_str)

        # Evaluate after every epoch
        curr_val_score = evaluate(
            dataloaders,
            task_cfg,
            device,
            model,
        )
        logger.info(
            f"[Validation] Current VQA: {curr_val_score} at {global_step} | Best VQA: {best_val_score} at {best_val_step}"
        )
        
        # temp code to output logging file
        logging_path = os.path.join(save_path, "eval_result.txt")
        with open(logging_path,'a') as log_file:
            log_file.write(f"VQA accuracy: {curr_val_score} at epoch {epoch_id + 1} | Best VQA: {best_val_score} at step {best_val_step}\n")
            log_file.write(f"lr rates = {[float(grp['lr']) for grp in optimizer.param_groups]}\n")
            
            
        if curr_val_score > best_val_score:
            logger.info(f"Saving Checkpoint: {checkpoint_path}")
            model_to_save = model.module if hasattr(model, "module") else model
            best_val_score, best_val_step = curr_val_score, global_step
            torch.save(
                {
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
                    "global_step": global_step,
                    "current_val_score": curr_val_score,
                    "epoch_id": epoch_id,
                },
                checkpoint_path,
            )

    print(
        f"Best Validation Score: {best_val_score}, Best Validation Epoch: {best_val_step}"
    )
    return checkpoint_path, model, dataloaders


def evaluate(
    dataloaders,
    task_cfg,
    device,
    model,
):
    scores, batch_sizes = [], []
    model.eval()
    with torch.no_grad():
        for batch_dict in tqdm(dataloaders["val"], desc="Validation"):
            loss, score, batch_size, _ = forward_model(
                task_cfg, device, model, batch_dict=batch_dict
            )
            scores.append(score * batch_size)
            batch_sizes.append(batch_size)

    model.train()
    return sum(scores) / sum(batch_sizes)


if __name__ == "__main__":
    
    # temp code for restoring checkpoint
    checkpoint_path, model, dataloaders, task_cfg, device = main()
    
    assert os.path.exists(checkpoint_path)
    task = registry["val_on"][0]
    # task = registry["test_on"][0]
    evaluator = Evaluator(checkpoint_path, model, dataloaders, task)

    
    for split in ["test", "val"]:
        evaluator.evaluate_no_beam(split=split)
        if split == "val":
            evaluator.evaluate_anls_no_beam(split=split)
    

