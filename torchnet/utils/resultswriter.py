import pickle


class ResultsWriter(object):
    '''Logs results to a file.

    The ResultsWriter provides a convenient interface for periodically writing
    results to a file. It is designed to capture all information for a given
    experiment, which may have a sequence of distinct tasks. Therefore, it writes
    results in the format::

        {
            'tasks': [...]
            'results': [...]
        }

    The ResultsWriter class chooses to use a top-level list instead of a dictionary
    to preserve temporal order of tasks (by default).

    Args:
        filepath (str): Path to write results to
        overwrite (bool): whether to clobber a file if it exists

    Example:
        >>> result_writer = ResultWriter(path)
        >>> for task in ['CIFAR-10', 'SVHN']:
        >>>    train_results = train_model()
        >>>    test_results = test_model()
        >>>    result_writer.update(task, {'Train': train_results, 'Test': test_results})

    '''

    def __init__(self, filepath, overwrite=False):
        if overwrite:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'tasks': [],
                    'results': []
                }, f)
        else:
            assert not os.path.exists(filepath), 'Cannot write results to "{}". Already exists!'.format(filepath)
        self.filepath = filepath
        self.tasks = set()

    def _add_task(self, task_name):
        assert task_name not in self.tasks, "Task already added! Use a different name."
        self.tasks.add(task_name)

    def update(self, task_name, result):
        ''' Update the results file with new information.

        Args:
            task_name (str): Name of the currently running task. A previously unseen
                ``task_name`` will create a new entry in both :attr:`tasks`
                and :attr:`results`.
            result: This will be appended to the list in :attr:`results` which
                corresponds to the ``task_name`` in ``task_name``:attr:`tasks`.

        '''
        with open(self.filepath, 'rb') as f:
            existing_results = pickle.load(f)
        if task_name not in self.tasks:
            self._add_task(task_name)
            existing_results['tasks'].append(task_name)
            existing_results['results'].append([])
        task_name_idx = existing_results['tasks'].index(task_name)
        results = existing_results['results'][task_name_idx]
        results.append(result)
        with open(self.filepath, 'wb') as f:
            pickle.dump(existing_results, f)
