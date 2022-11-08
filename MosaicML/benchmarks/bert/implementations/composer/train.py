# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import tempfile
import warnings

from composer.trainer.trainer_hparams import TrainerHparams, model_registry
from composer.utils import dist

# register BERTUnpadded model
from bert_hparams import BERTUnpaddedHparams
model_registry['bert_unpadded'] = BERTUnpaddedHparams


def is_performance_installed():
    try:
        import performance as perf
        del  perf
    except ModuleNotFoundError:
        return False
    else:
        return True

if is_performance_installed():
    import performance


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'


def main() -> None:
    warnings.formatwarning = warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    if is_performance_installed():
        performance.register_all_algorithms()

    hparams = TrainerHparams.create(
        cli_args=True)  # reads cli args from sys.argv

    trainer = hparams.initialize_object()

    # Only log the config once, since it should be the same on all ranks.
    if dist.get_global_rank() == 0:
        with tempfile.TemporaryDirectory() as tmpdir:
            hparams_name = os.path.join(tmpdir, 'hparams.yaml')
            with open(hparams_name, 'w+') as f:
                f.write(hparams.to_yaml())
            trainer.logger.upload_file(
                remote_file_name=f"{trainer.state.run_name}/hparams.yaml",
                file_path=f.name,
                overwrite=True)

    # Print the config to the terminal
    if dist.get_local_rank() == 0:
        print("*" * 30)
        print("Config:")
        print(hparams.to_yaml())
        print("*" * 30)

    trainer.fit()


if __name__ == "__main__":
    main()
