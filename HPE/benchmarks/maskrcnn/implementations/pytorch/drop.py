from maskrcnn_benchmark.utils.mlperf_logger import mllogger
mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True, stack_offset=1)
