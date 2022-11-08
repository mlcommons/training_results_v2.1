# import collections
# import os
# import subprocess

import os

from mlperf_logging import mllog
from mlperf_logging.mllog import constants

from mlperf_common.logging import MLLoggerWrapper
from mlperf_common.frameworks.pyt import PyTCommunicationHandler


class MLLoggerWrapperCloud(MLLoggerWrapper):
    """Set division to CLOUD instead of ONPREM
    """

    def mlperf_submission_log(self, benchmark, num_nodes=None, org=None,
                              platform=None):
        """ Helper for logging submission entry. """
        if num_nodes is None:
            num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', 1)

        if org is None:
            org = os.environ.get('MLPERF_SUBMISSION_ORG',
                                 'SUBMISSION_ORG_PLACEHOLDER')

        if platform is None:
            platform = os.environ.get('MLPERF_SUBMISSION_PLATFORM',
                                      'SUBMISSION_PLATFORM_PLACEHOLDER')

        self.event(
            key=constants.SUBMISSION_BENCHMARK,
            value=benchmark,
            internal_call=True)

        self.event(
            key=constants.SUBMISSION_ORG,
            value=org,
            internal_call=True)

        self.event(
            key=constants.SUBMISSION_DIVISION,
            value=constants.CLOSED,
            internal_call=True)

        self.event(
            key=constants.SUBMISSION_STATUS,
            value=constants.CLOUD,
            internal_call=True)

        self.event(
            key=constants.SUBMISSION_PLATFORM,
            value=f'{num_nodes}x{platform}',
            internal_call=True)


mllogger = MLLoggerWrapperCloud(PyTCommunicationHandler())
