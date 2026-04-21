# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from transformers.video_utils import VideoMetadata
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image
from ...utils.multimodal_contract import load_video_tensors_and_metadata
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray, list], repeats: int) -> Union[torch.Tensor, np.ndarray, list]:
    # repeat the elements, supports tensor, numpy array and list
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    elif isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)
    elif isinstance(value, list):
        out = []
        for v in value:
            out.extend([v] * repeats)
        return out
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output vision special tokens (image/video placeholders)
    if processor is None:
        return None

    logit_bias = {}
    if hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        logit_bias[image_token_id] = -100
    if hasattr(processor, "video_token"):
        video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
        logit_bias[video_token_id] = -100

    return logit_bias if logit_bias else None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any],
    image_min_pixels: int,
    image_max_pixels: int,
    video_min_pixels: int,
    video_max_pixels: int,
    video_max_frames: int,
    video_fps: float,
    video_total_pixels: Optional[int],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    images, videos = [], []
    mm_kwargs = None

    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, image_min_pixels, image_max_pixels))

    video_tensors, video_metadatas = load_video_tensors_and_metadata(
        multi_modal_data,
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
        video_max_frames=video_max_frames,
        video_fps=video_fps,
        video_total_pixels=video_total_pixels,
    )
    if video_tensors:
        if video_metadatas is None or len(video_metadatas) != len(video_tensors):
            metadata_count = 0 if video_metadatas is None else len(video_metadatas)
            raise ValueError(
                "Resolved video data is missing valid metadata entries. "
                f"Got {len(video_tensors)} video tensors and {metadata_count} metadata entries."
            )
        for tensor, metadata in zip(video_tensors, video_metadatas):
            videos.append((tensor, metadata))
        mm_kwargs = {"do_sample_frames": False, "do_resize": False}

    if len(images) != 0:
        return {"image": images}, None

    if len(videos) != 0:
        return {"video": videos}, mm_kwargs

    return None, None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.tokenizer = tokenizer  # 保存 tokenizer 用于 mix-policy 离线轨迹处理
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)

        # Mix-policy 配置
        self.enable_mix_policy = getattr(config, 'enable_mix_policy', False)
        self.mix_policy_accuracy_threshold = getattr(config, "mix_policy_accuracy_threshold", None)
        if self.enable_mix_policy and self.rank == 0:
            if self.mix_policy_accuracy_threshold is None:
                print("Mix-policy enabled: offline trajectories will replace last online generation when available.")
            else:
                print(
                    "Mix-policy enabled with trainer-side conditional gating: "
                    f"offline is injected only when online group accuracy < {self.mix_policy_accuracy_threshold}."
                )
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images, "video": 1}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params = self.sampling_params
        if kwargs:
            # vLLM 0.17+: SamplingParams is a msgspec.Struct with read-only properties
            # Get the valid constructor fields
            struct_fields = getattr(self.sampling_params, "__struct_fields__", None)
            if struct_fields is not None:
                # msgspec.Struct: rebuild with current values + overrides
                current_args = {}
                for field in struct_fields:
                    if not field.startswith("_"):
                        current_args[field] = getattr(self.sampling_params, field)
                for key, value in kwargs.items():
                    if key in current_args:
                        current_args[key] = value
                self.sampling_params = SamplingParams(**current_args)
            else:
                # Older vLLM: use setattr directly
                for key, value in kwargs.items():
                    if hasattr(self.sampling_params, key):
                        setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        self.sampling_params = old_sampling_params

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)

        # Mix-policy: 获取预采集轨迹信息
        batch_has_offline = non_tensor_batch.pop("has_offline_trajectory", None)
        batch_offline_output = non_tensor_batch.pop("offline_output", None)

        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                item = {"prompt_token_ids": list(raw_prompt_ids)}
                mm_data, mm_kwargs = _process_multi_modal_data(
                    multi_modal_data,
                    prompts.meta_info["image_min_pixels"],
                    prompts.meta_info["image_max_pixels"],
                    prompts.meta_info["video_min_pixels"],
                    prompts.meta_info["video_max_pixels"],
                    prompts.meta_info["video_max_frames"],
                    prompts.meta_info["video_fps"],
                    prompts.meta_info.get("video_total_pixels"),
                )
                if mm_data is not None:
                    if "video" in mm_data:
                        videos = []
                        for tensor, metadata in mm_data["video"]:
                            if isinstance(metadata, dict):
                                metadata_obj = VideoMetadata(
                                    total_num_frames=metadata.get("total_num_frames", tensor.shape[0] if hasattr(tensor, "shape") else len(tensor)),
                                    fps=metadata.get("fps"),
                                    frames_indices=metadata.get("frames_indices"),
                                    video_backend=metadata.get("video_backend"),
                                    width=metadata.get("width"),
                                    height=metadata.get("height"),
                                    duration=metadata.get("duration"),
                                )
                            else:
                                metadata_obj = metadata
                            videos.append((tensor, metadata_obj))
                        item["multi_modal_data"] = {"video": videos}
                    else:
                        item["multi_modal_data"] = mm_data
                    if mm_kwargs is not None:
                        item["mm_processor_kwargs"] = mm_kwargs
                vllm_inputs.append(item)
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

            # Mix-policy: 用离线轨迹替换有预采集数据样本的最后一个在线生成
            # 对于没有离线轨迹的样本，保留所有 n 个在线生成
            if (
                self.enable_mix_policy
                and self.mix_policy_accuracy_threshold is None
                and batch_has_offline is not None
                and self.sampling_params.n > 1
            ):
                response_ids = self._mix_offline_trajectories(
                    response_ids=response_ids,
                    batch_has_offline=batch_has_offline,
                    batch_offline_output=batch_offline_output,
                    n=self.sampling_params.n,
                )

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    def _mix_offline_trajectories(
        self,
        response_ids: torch.Tensor,
        batch_has_offline: np.ndarray,
        batch_offline_output: np.ndarray,
        n: int,
    ) -> torch.Tensor:
        """
        Mix-policy 核心逻辑: 用离线轨迹替换有预采集数据样本的最后一个在线生成。

        对于有离线轨迹的样本: 在线生成 n-1 个 + 1 个离线 = n 个总样本
        对于没有离线轨迹的样本: 在线生成 n 个 = n 个总样本

        处理逻辑与原始 rollout 一致:
        1. Tokenize 离线轨迹得到 token list
        2. 使用 VF.pad_2d_list_to_length 进行 padding (与在线生成的处理方式一致)
        3. 后续的 EOS/mask 处理由 VF.get_response_mask 统一完成

        Args:
            response_ids: 在线生成的 response ids, shape (batch_size * n, response_length)
            batch_has_offline: 每个原始样本是否有离线轨迹, shape (original_batch_size,)
            batch_offline_output: 每个原始样本的离线输出文本, shape (original_batch_size,)
            n: 每个样本的生成数量

        Returns:
            混合后的 response_ids, shape (batch_size * n, response_length)
        """
        device = response_ids.device
        response_length = response_ids.size(1)
        original_batch_size = len(batch_has_offline)

        # 遍历每个原始样本
        for i in range(original_batch_size):
            # 检查该样本是否有离线轨迹
            has_offline = batch_has_offline[i] if batch_has_offline is not None else False
            offline_output = batch_offline_output[i] if batch_offline_output is not None else ""

            if has_offline and offline_output and len(offline_output.strip()) > 0:
                # 1. Tokenize 离线轨迹 (与 vLLM 生成的 token_ids 格式一致)
                offline_token_ids = self.tokenizer.encode(
                    offline_output,
                    add_special_tokens=False,
                )

                # 2. 使用与原始代码相同的 padding 方式
                # VF.pad_2d_list_to_length 接受 2D list，这里包装成单样本的 2D list
                offline_tokens = VF.pad_2d_list_to_length(
                    [offline_token_ids],  # 包装成 2D list: [[token1, token2, ...]]
                    self.pad_token_id,
                    max_length=response_length,
                ).to(device)
                # offline_tokens shape: (1, response_length)

                # 3. 计算要替换的位置：第 i 个样本的最后一个生成 (即第 n-1 个)
                # response_ids 的布局是: [sample_0_gen_0, sample_0_gen_1, ..., sample_0_gen_{n-1},
                #                         sample_1_gen_0, sample_1_gen_1, ..., sample_1_gen_{n-1}, ...]
                replace_idx = i * n + (n - 1)  # 替换最后一个在线生成
                response_ids[replace_idx] = offline_tokens.squeeze(0)

        # 后续的 EOS/mask 处理由 generate_sequences 中的 VF.get_response_mask 统一完成
        return response_ids
