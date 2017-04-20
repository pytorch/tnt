import torch

def model_and_optim(network, optimizer, path):
    hooks = {}

    def on_end_epoch(state):
        print('Saving Checkpoint...', end='')
        torch.save({
            'network': network,
            'optimizer': optimizer,
            'epoch': state['epoch']}, path)
        print(' - Done')


    hooks['on_end_epoch'] = on_end_epoch
    return hooks
