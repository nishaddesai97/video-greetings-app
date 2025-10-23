workers = 1  # Reduce number of workers due to memory constraints
worker_class = 'sync'
timeout = 300  # Increase timeout for video processing
max_requests = 1
max_requests_jitter = 1
preload_app = True
worker_tmp_dir = '/tmp'
