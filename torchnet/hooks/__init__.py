from . import classification
from . import checkpoint
from . import generative

def add_hooks(engine, hooks):
    for name, hook in hooks.items():
        if name in engine.hooks:
            old_hook = engine.hooks[name]
            def new_hook(state):
                old_hook(state)
                hook(state)
            engine.hooks[name] = new_hook
        else:
            engine.hooks[name] = hook

def replace_hooks(engine, hooks):
    engine.hooks = hooks
