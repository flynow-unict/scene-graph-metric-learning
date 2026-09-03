import concurrent.futures
def fetch_with_timeout(func, *args, timeout=10, **kwargs):
    """Esegue una funzione con un timeout espresso in secondi."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None