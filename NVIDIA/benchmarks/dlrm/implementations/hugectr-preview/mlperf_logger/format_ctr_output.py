# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import mlperf_logging.mllog as mllog
import mlperf_logging.mllog.constants as mlperf_constants

HCTR_LOG_MSG = (
    "^\\[HCTR\\]\\[\\d{2}:\\d{2}:\\d{2}\\.\\d{3}\\]"
    "\\[[A-Z]+\\]\\[RK\\d\\]\\[main\\]: (?P<message>.*)$"
)
HCTR_HIT_TARGET_MSG = (
    "^Hit target accuracy AUC \\d\\.\\d+ "
    "at (?P<hit_iter>\\d+) / (?P<max_iter>\\d+) iterations "
    "with batchsize \\d+ in \\d+\\.\\d{2}s. "
    "Average speed (?P<throughput>\\d+\\.\\d+) records/s.$"
)

# layer names used in HugeCTR configs mapped to corresponding MLPerf names
HCTR_TO_MLPERF_LAYER_NAME = {
    "sparse_embedding1": "embeddings",
    "fc11-fc12-fc13-fc14": "bottom_mlp_dense1",
    "fc21-fc22-fc23-fc24": "bottom_mlp_dense2",
    "fc3": "bottom_mlp_dense3",
    "fc41-fc42-fc43-fc44": "top_mlp_dense1",
    "fc51-fc52-fc53-fc54": "top_mlp_dense2",
    "fc61-fc62-fc63-fc64": "top_mlp_dense3",
    "fc71-fc72-fc73-fc74": "top_mlp_dense4",
    "fc8": "top_mlp_dense5",
}

mllogger = mllog.get_mllogger()


def log_submission(benchmark, time_ms=None):
    num_nodes = os.environ.get("SLURM_JOB_NUM_NODES", "1")

    mllogger.event(
        time_ms=time_ms,
        key=mlperf_constants.SUBMISSION_BENCHMARK,
        value=benchmark,
    )
    mllogger.event(
        time_ms=time_ms,
        key=mlperf_constants.SUBMISSION_ORG,
        value="NVIDIA",
    )
    mllogger.event(
        time_ms=time_ms,
        key=mlperf_constants.SUBMISSION_DIVISION,
        value="closed",
    )
    mllogger.event(
        time_ms=time_ms,
        key=mlperf_constants.SUBMISSION_STATUS,
        value="onprem",
    )
    mllogger.event(
        time_ms=time_ms,
        key=mlperf_constants.SUBMISSION_PLATFORM,
        value="{}xSUBMISSION_PLATFORM_PLACEHOLDER".format(num_nodes),
    )


def log_hparams(config, time_ms=None):
    mllogger.event(
        time_ms=time_ms,
        key="eval_samples",
        value=config["eval_num_samples"],
    )
    mllogger.event(
        time_ms=time_ms,
        key="global_batch_size",
        value=config["global_batch_size"],
    )
    mllogger.event(
        time_ms=time_ms,
        key="opt_base_learning_rate",
        value=config["opt_base_learning_rate"],
    )
    mllogger.event(
        time_ms=time_ms,
        key="sgd_opt_base_learning_rate",
        value=config["sgd_opt_base_learning_rate"],
    )
    mllogger.event(
        time_ms=time_ms,
        key="sgd_opt_learning_rate_decay_poly_power",
        value=config["sgd_opt_learning_rate_decay_poly_power"],
    )
    mllogger.event(
        time_ms=time_ms,
        key="opt_learning_rate_warmup_steps",
        value=config["opt_learning_rate_warmup_steps"],
    )
    mllogger.event(
        time_ms=time_ms,
        key="opt_learning_rate_warmup_factor",
        value=config["opt_learning_rate_warmup_factor"],
    )
    mllogger.event(
        time_ms=time_ms,
        key="lr_decay_start_steps",
        value=config["lr_decay_start_steps"],
    )
    mllogger.event(
        time_ms=time_ms,
        key="sgd_opt_learning_rate_decay_steps",
        value=config["sgd_opt_learning_rate_decay_steps"],
    )
    mllogger.event(
        time_ms=time_ms,
        key="gradient_accumulation_steps",
        value=config["gradient_accumulation_steps"],
    )


def log_config(config: Dict[str, Any], time_ms: int = None):
    # print hparams and submission info on the first node only
    if os.environ.get("SLURM_NODEID", "0") == "0":
        log_submission("dlrm", time_ms)
        log_hparams(config, time_ms)


@dataclass
class HugectrLog:
    time_ms: int
    key: str
    value: Any = None
    metadata: Dict[str, Any] = None

    @classmethod
    def from_raw_log(cls, msg: str, ignore_errors: bool = False):
        try:
            time_ms, key, *data = msg.strip("[], \n").split(", ")
            time_ms = round(float(time_ms))
            assert hasattr(mlperf_constants, key.upper())
            value = cls._get_value(key, data)
            metadata = cls._get_metadata(key, data)
        except:
            if ignore_errors:
                return None
            else:
                raise
        else:
            return cls(time_ms, key, value, metadata)

    @classmethod
    def _get_value(cls, key: str, data: List[str]):
        if key == mlperf_constants.EVAL_ACCURACY:
            return float(data[0])
        if key == mlperf_constants.TRAIN_SAMPLES:
            return int(data[0])

    @classmethod
    def _get_metadata(cls, key: str, data: List[str]):
        if key == mlperf_constants.WEIGHTS_INITIALIZATION:
            return {"tensor": HCTR_TO_MLPERF_LAYER_NAME[data[0]]}
        if key == mlperf_constants.EVAL_ACCURACY:
            return {"epoch_num": float(data[1]) + 1.0}
        if key.startswith("eval"):
            return {"epoch_num": float(data[0]) + 1.0}
        if key.startswith("epoch"):
            return {"epoch_num": int(data[0]) + 1}
        if key == mlperf_constants.RUN_STOP:
            # status should be overridden if the run turns out
            # to be successful based on other log messages
            return {"status": "aborted"}


class LogConverter:
    def __init__(self, start_time: int):
        self.start_time = start_time
        self._last_eval_accuracy = -1.0

    def _process(self, hugectr_log: HugectrLog):
        """Analyse a log, modify if needed, and conclude whether it should be reported."""
        hugectr_log.time_ms += self.start_time
        if hugectr_log.key == mlperf_constants.EVAL_ACCURACY:
            self._last_eval_accuracy = hugectr_log.value
        elif hugectr_log.key == mlperf_constants.RUN_STOP and self._last_eval_accuracy > 0.8025:
            hugectr_log.metadata = {"status": "success"}

    def _get_log_method(self, key: str):
        if key.endswith("_start"):
            return mllogger.start
        elif key.endswith("_stop"):
            return mllogger.end
        return mllogger.event

    def log_event(self, msg: str):
        hugectr_log = HugectrLog.from_raw_log(msg, ignore_errors=True)
        if hugectr_log is not None:
            self._process(hugectr_log)
            log_method = self._get_log_method(hugectr_log.key)
            log_method(
                key=hugectr_log.key,
                value=hugectr_log.value,
                metadata=hugectr_log.metadata,
                time_ms=hugectr_log.time_ms,
            )

    def log_tracked_stats(self, line: str):
        """Read additional statistics from the final log line."""
        if (stats := re.match(HCTR_HIT_TARGET_MSG, line)) is not None:
            mllogger.event(
                key="tracked_stats",
                value={"throughput": float(stats["throughput"])},
                metadata={"step": 1.0 + round(int(stats["hit_iter"]) / int(stats["max_iter"]), 6)},
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to the logs to be translated"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="HugeCTR input config file in JSON format",
    )

    parser.add_argument(
        "--start_time",
        type=int,
        required=True,
        help="Training start time as the seconds elapsed since UNIX Epoch",
    )
    args = parser.parse_args()

    # Convert to ms to be consistent with the MLPerf logging API
    start_time_ms = args.start_time * 1000

    with open(args.config_file, "r") as f:
        config = json.load(f)
    log_config(config, start_time_ms)

    converter = LogConverter(start_time=start_time_ms)

    with open(args.log_path, errors="ignore") as f:
        log_lines = f.readlines()

    for line in log_lines:
        if (log_match := re.match(HCTR_LOG_MSG, line)) is not None:
            converter.log_event(log_match["message"])
            converter.log_tracked_stats(log_match["message"])


if __name__ == "__main__":
    main()
