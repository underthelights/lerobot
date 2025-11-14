#!/usr/bin/env python

# Copyright 2024
# Modified by LG AI Research Vision Lab
# Added MLflow + CSV logging + file logging + resource logging + artifacts sync

import logging
import time
import csv
import os
from contextlib import nullcontext
from pprint import pformat
from typing import Any
from datetime import datetime
from pathlib import Path

import psutil
import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

# ==== MLflow (optional) ====
try:
    import mlflow
    _MLFLOW = True
except ImportError:
    mlflow = None
    _MLFLOW = False
# ===========================

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval import eval_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    한 step 학습 업데이트 (loss backward, grad clip, optimizer step, scheduler step)
    """
    start = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()

    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)

    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    with lock if lock else nullcontext():
        grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()

    if lr_scheduler:
        lr_scheduler.step()

    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start

    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    """
    메인 학습 루프 (MLflow + CSV + 파일로그 + 리소스 로깅 포함)
    """

    # ----- logging 기본 세팅 (console + 이후 file handler 추가) -----
    init_logging()
    logging.info(f"[DEBUG] _MLFLOW flag = {_MLFLOW}")

    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    # ===========================
    # MLflow Setup (Local SQLite + File Artifacts)
    # ===========================
    if _MLFLOW:
        # outputs_root = .../lerobot/outputs (컨테이너 기준)
        outputs_root = cfg.output_dir.resolve().parents[1]
        sqlite_db_path = outputs_root / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{sqlite_db_path.as_posix()}")

        tracking_dir = outputs_root / "mlruns"  # artifacts root
        artifact_uri = f"file:{tracking_dir.as_posix()}"

        exp_name = cfg.job_name or "lerobot_training"
        run_name = Path(cfg.output_dir).name

        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            client.create_experiment(
                name=exp_name,
                artifact_location=artifact_uri,
            )
        mlflow.set_experiment(exp_name)
        mlflow.start_run(run_name=run_name)

        # 디버깅용 경로 로그
        logging.info(f"[MLflow] tracking_uri = {mlflow.get_tracking_uri()}")
        logging.info(f"[MLflow] sqlite_db_path = {sqlite_db_path}")
        logging.info(f"[MLflow] artifact_uri  = {artifact_uri}")
        logging.info(f"[MLflow] experiment = {exp_name}, run_name = {run_name}")

        mlflow.set_tag("output_dir", str(cfg.output_dir))
        mlflow.set_tag("policy_type", cfg.policy.type)
        mlflow.set_tag("dataset", getattr(cfg.dataset, "repo_id", None))

        mlflow.log_params(
            {
                "policy_type": cfg.policy.type,
                "dataset": getattr(cfg.dataset, "repo_id", None),
                "steps": cfg.steps,
                "batch_size": cfg.batch_size,
                "lr": cfg.optimizer.lr,
            }
        )

    # ================
    # CSV LOGGING SETUP
    # ================
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model_name = cfg.policy.type  # act, diffusion, pi0, pi0fast, smolvla 등
    date_str = datetime.now().strftime("%Y%m%d")
    csv_file = cfg.output_dir / f"training_log_{model_name}_{date_str}.csv"
    csv_header_written = False

    logging.info(f"[CSV] training log will be written to: {csv_file}")

    # ================
    # FILE LOGGING (train.log)
    # ================
    log_file = cfg.output_dir / "train.log"
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
        logging.info(f"[LOG] file logging enabled: {log_file}")
    except Exception as e:
        logging.warning(f"[LOG] failed to set file handler: {e}")

    # W&B 그대로 유지
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs saved locally.", "yellow", attrs=["bold"]))

    # seed
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # device
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset...")
    dataset = make_dataset(cfg)

    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating eval env...")
        eval_env = make_env(
            cfg.env,
            n_envs=cfg.eval.batch_size,
            use_async_envs=cfg.eval.use_async_envs,
        )

    logging.info("Creating policy...")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)

    logging.info("Creating optimizer...")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    # Stats
    num_learnable_params = sum(
        p.numel() for p in policy.parameters() if p.requires_grad
    )
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(f"Output dir: {cfg.output_dir}")
    logging.info(f"Steps: {cfg.steps} ({format_big_number(cfg.steps)})")
    logging.info(f"Dataset frames: {dataset.num_frames}")
    logging.info(f"Episodes: {dataset.num_episodes}")
    logging.info(f"Params: {num_learnable_params}/{num_total_params}")

    # Dataloader
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    # Metrics
    metric_dict = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        metric_dict,
        initial_step=step,
    )

    logging.info("Start training...")

    # =====================
    # 메인 학습 루프
    # =====================
    for _ in range(step, cfg.steps):
        start = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start

        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(
                    device, non_blocking=(device.type == "cuda")
                )

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler,
            lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        step += 1
        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        is_save_step = cfg.save_freq > 0 and step % cfg.save_freq == 0

        # ========= LOGGING =========
        if is_log_step:
            # ----- 리소스 측정 (GPU/CPU 메모리) -----
            gpu_mem_mb = None
            if device.type == "cuda" and torch.cuda.is_available():
                try:
                    cur_dev = torch.cuda.current_device()
                    gpu_mem_mb = torch.cuda.memory_allocated(cur_dev) / (1024**2)
                except Exception:
                    gpu_mem_mb = None

            proc = psutil.Process(os.getpid())
            cpu_mem_mb = proc.memory_info().rss / (1024**2)

            # 콘솔 + train.log 로깅
            if gpu_mem_mb is not None:
                logging.info(
                    f"{train_tracker} | gpu_mem={gpu_mem_mb:.1f}MB | cpu_mem={cpu_mem_mb:.1f}MB"
                )
            else:
                logging.info(
                    f"{train_tracker} | cpu_mem={cpu_mem_mb:.1f}MB"
                )

            # MLflow
            if _MLFLOW:
                mlflow.log_metric("loss", train_tracker.loss.avg, step)
                mlflow.log_metric("grad_norm", train_tracker.grad_norm.avg, step)
                mlflow.log_metric("lr", train_tracker.lr.avg, step)
                mlflow.log_metric("cpu_mem_mb", cpu_mem_mb, step)
                if gpu_mem_mb is not None:
                    mlflow.log_metric("gpu_mem_mb", gpu_mem_mb, step)

            # CSV Logging
            try:
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    if not csv_header_written:
                        writer.writerow(
                            [
                                "step",
                                "loss",
                                "grad_norm",
                                "lr",
                                "update_s",
                                "data_s",
                                "gpu_mem_mb",
                                "cpu_mem_mb",
                            ]
                        )
                        csv_header_written = True
                    writer.writerow(
                        [
                            step,
                            train_tracker.loss.avg,
                            train_tracker.grad_norm.avg,
                            train_tracker.lr.avg,
                            train_tracker.update_s.avg,
                            train_tracker.dataloading_s.avg,
                            gpu_mem_mb,
                            cpu_mem_mb,
                        ]
                    )
            except Exception as e:
                logging.warning(f"[CSV] write failed: {e}")

            # W&B
            if wandb_logger:
                wandb_logger.log_dict(train_tracker.to_dict(), step)

            train_tracker.reset_averages()

        # ========= SAVE CHECKPOINT =========
        if cfg.save_checkpoint and is_save_step:
            ckpt_dir = get_step_checkpoint_dir(
                cfg.output_dir, cfg.steps, step
            )
            save_checkpoint(
                ckpt_dir, step, cfg, policy, optimizer, lr_scheduler
            )
            update_last_checkpoint(ckpt_dir)
            if wandb_logger:
                wandb_logger.log_policy(ckpt_dir)

        # ========= EVAL =========
        if eval_env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            with torch.no_grad(), (
                torch.autocast(device_type=device.type)
                if cfg.policy.use_amp
                else nullcontext()
            ):
                eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir
                    / "eval"
                    / f"videos_step_{step_id}",
                )

    if eval_env:
        eval_env.close()

    logging.info("Training finished.")

    # ========= MLflow: output_dir 전체를 artifacts로 복제 =========
    if _MLFLOW:
        try:
            mlflow.log_artifacts(str(cfg.output_dir), artifact_path="run_outputs")
            logging.info(
                f"[MLflow] logged artifacts from {cfg.output_dir} to 'run_outputs/' in this run."
            )
        except Exception as e:
            logging.warning(f"[MLflow] failed to log artifacts: {e}")

        mlflow.end_run()

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    train()
