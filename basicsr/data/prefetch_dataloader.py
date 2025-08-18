import threading
import jittor as jt
import queue as Queue


class PrefetchGenerator(threading.Thread):
    """A general prefetch generator.

    Ref:
    https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
        generator: Python generator.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, generator, num_prefetch_queue):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader:
    """Prefetch version of dataloader.

    Ref:
    https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

    TODO:
    Need to test on single gpu and ddp (multi-gpu). There is a known issue in
    ddp.

    Args:
        num_prefetch_queue (int): Number of prefetch queue.
        kwargs (dict): Other arguments for dataloader.
    """

    def __init__(self, num_prefetch_queue, **kwargs):
        self.num_prefetch_queue = num_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)


class CPUPrefetcher:
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)


class CUDAPrefetcher:
    """Jittor prefetcher (no CUDA streams; same API as your PyTorch version).

    Args:
        loader: an iterable dataloader
        opt (dict): expects opt["num_gpu"] (>=1 -> use cuda)
    """

    def __init__(self, loader, opt):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt

        # Jittor uses global flags to select device
        use_cuda = bool(opt.get("num_gpu", 0)) and jt.has_cuda
        if use_cuda:
            jt.flags.use_cuda = 1
            self.device = "cuda"
        else:
            self.device = "cpu"

        self._preload()

    # -------- internals --------
    def _to_var(self, x):
        # If it's already a jt.Var, return it; otherwise try to convert (numpy/list/number)
        if isinstance(x, jt.Var):
            return x
        try:
            return jt.array(x)
        except Exception:
            # leave non-array types (e.g., strings, dicts with metadata) untouched
            return x

    def _move_batch_to_device(self, batch):
        # Support dict / list / tuple / single tensor
        if isinstance(batch, dict):
            return {k: self._to_var(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            t = type(batch)
            return t(self._to_var(v) for v in batch)
        else:
            return self._to_var(batch)

    def _preload(self):
        try:
            b = next(self.loader)  # may be dict/list/tuple
        except StopIteration:
            self.batch = None
            return
        self.batch = self._move_batch_to_device(b)

    # -------- public API (same as your original) --------
    def next(self):
        batch = self.batch
        self._preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self._preload()
