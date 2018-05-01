import pickle

class ResultsWriter(object):
    ''' 
        Logs results to a file.
        
        Stores in the format:
            {
                'tasks': [...]
                'results': [...]
            }
        We use lists instead of a dictionary to preserve temporal order of tasks (by default)

        Example:
            result_writer = ResultWriter(path)
            for task in ['CIFAR-10', 'SVHN']:
                train_results = train_model()
                test_results = test_model()
                result_writer.update(task, {'Train': train_results, 'Test': test_results})
    '''

    def __init__(self, filepath, overwrite=False):
        '''
            Args:
                filepath: Path to use
                overwrite: bool, whether to clobber a file if it exists

        '''
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
        '''
            Args:
                task_name: Name of the currently running task/experiment
                result: Result to append to the currently running experiment 
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
                