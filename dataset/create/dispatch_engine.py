from typing import Optional


class AsyncDPDispathEngine:
    def __init__(self, dataset_path: Optional[str] = None, dataset_split: Optional[str] = None, data_files: Optional[str] = None):
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.data_files = data_files

        self.task_dict = {}
        self.result_queue = []
        self.next_task_id = 0

        self.dataset = self.build_dataset()
        self.dataset_iter = iter(self.dataset)

        self.still_running = True

        self.workers = []

    def preprocess(self, sample):
        return sample

    def build_dataset(self):
        raise NotImplementedError

    def _make_task(self, sample):
        raise NotImplementedError
    
    def fetch_task(self):
        if not self.still_running:
            return None
        try:
            sample = next(self.dataset_iter)
            sample = self.preprocess(sample)
            task_id = self.next_task_id
            self.next_task_id += 1
            task = self._make_task(sample)
            task["task_id"] = task_id
            self.task_dict[task_id] = task
            return task
        except Exception as e:
            self.still_running = False
            if isinstance(e, StopIteration):
                print("Reached the end of the dataset.")
            else:
                print(e.with_traceback())
            return None

    def register(self, id: str):
        if id not in self.workers:
            self.workers.append(id)
    
    def unregister(self, id: str):
        if id in self.workers:
            self.workers.remove(id)
