""" Logging to Visdom server """
import numpy as np
import visdom

from .logger import Logger


class BaseVisdomLogger(Logger):
    '''
        The base class for logging output to Visdom.

        ***THIS CLASS IS ABSTRACT AND MUST BE SUBCLASSED***

        Note that the Visdom server is designed to also handle a server architecture,
        and therefore the Visdom server must be running at all times. The server can
        be started with
        $ python -m visdom.server
        and you probably want to run it from screen or tmux.
    '''

    @property
    def viz(self):
        return self._viz

    def __init__(self, fields=None, win=None, env=None, opts={}, port=8097, server="localhost"):
        super(BaseVisdomLogger, self).__init__(fields)
        self.win = win
        self.env = env
        self.opts = opts
        self._viz = visdom.Visdom(server="http://" + server, port=port)

    def log(self, *args, **kwargs):
        raise NotImplementedError(
            "log not implemented for BaseVisdomLogger, which is an abstract class.")

    def _viz_prototype(self, vis_fn):
        ''' Outputs a function which will log the arguments to Visdom in an appropriate way.

            Args:
                vis_fn: A function, such as self.vis.image
        '''
        def _viz_logger(*args, **kwargs):
            self.win = vis_fn(*args,
                              win=self.win,
                              env=self.env,
                              opts=self.opts,
                              **kwargs)
        return _viz_logger

    def log_state(self, state):
        """ Gathers the stats from self.trainer.stats and passes them into
            self.log, as a list """
        results = []
        for field_idx, field in enumerate(self.fields):
            parent, stat = None, state
            for f in field:
                parent, stat = stat, stat[f]
            results.append(stat)
        self.log(*results)


class VisdomSaver(object):
    ''' Serialize the state of the Visdom server to disk.
        Unless you have a fancy schedule, where different are saved with different frequencies,
        you probably only need one of these.
    '''

    def __init__(self, envs=None, port=8097, server="localhost"):
        super(VisdomSaver, self).__init__()
        self.envs = envs
        self.viz = visdom.Visdom(server="http://" + server, port=port)

    def save(self, *args, **kwargs):
        self.viz.save(self.envs)


class VisdomLogger(BaseVisdomLogger):
    '''
        A generic Visdom class that works with the majority of Visdom plot types.
    '''

    def __init__(self, plot_type, fields=None, win=None, env=None, opts={}, port=8097, server="localhost"):
        '''
            Args:
                fields: Currently unused
                plot_type: The name of the plot type, in Visdom

            Examples:
                >>> # Image example
                >>> img_to_use = skimage.data.coffee().swapaxes(0,2).swapaxes(1,2)
                >>> image_logger = VisdomLogger('image')
                >>> image_logger.log(img_to_use)

                >>> # Histogram example
                >>> hist_data = np.random.rand(10000)
                >>> hist_logger = VisdomLogger('histogram', , opts=dict(title='Random!', numbins=20))
                >>> hist_logger.log(hist_data)
        '''
        super(VisdomLogger, self).__init__(fields, win, env, opts, port, server)
        self.plot_type = plot_type
        self.chart = getattr(self.viz, plot_type)
        self.viz_logger = self._viz_prototype(self.chart)

    def log(self, *args, **kwargs):
        self.viz_logger(*args, **kwargs)


class VisdomPlotLogger(BaseVisdomLogger):

    def __init__(self, plot_type, fields=None, win=None, env=None, opts={}, port=8097, server="localhost", name=None):
        '''
            Multiple lines can be added to the same plot with the "name" attribute (see example)
            Args:
                fields: Currently unused
                plot_type: {scatter, line}

            Examples:
                >>> scatter_logger = VisdomPlotLogger('line')
                >>> scatter_logger.log(stats['epoch'], loss_meter.value()[0], name="train")
                >>> scatter_logger.log(stats['epoch'], loss_meter.value()[0], name="test")
        '''
        super(VisdomPlotLogger, self).__init__(fields, win, env, opts, port, server)
        valid_plot_types = {
            "scatter": self.viz.scatter,
            "line": self.viz.line}
        self.plot_type = plot_type
        # Set chart type
        if plot_type not in valid_plot_types.keys():
            raise ValueError("plot_type \'{}\' not found. Must be one of {}".format(
                plot_type, valid_plot_types.keys()))
        self.chart = valid_plot_types[plot_type]

    def log(self, *args, **kwargs):
        if self.win is not None and self.viz.win_exists(win=self.win, env=self.env):
            if len(args) != 2:
                raise ValueError("When logging to {}, must pass in x and y values (and optionally z).".format(
                    type(self)))
            x, y = args
            self.chart(
                X=np.array([x]),
                Y=np.array([y]),
                update='append',
                win=self.win,
                env=self.env,
                opts=self.opts,
                **kwargs)
        else:
            if self.plot_type == 'scatter':
                chart_args = {'X': np.array([args])}
            else:
                chart_args = {'X': np.array([args[0]]),
                              'Y': np.array([args[1]])}
            self.win = self.chart(
                win=self.win,
                env=self.env,
                opts=self.opts,
                **chart_args)
            # For some reason, the first point is a different trace. So for now
            # we can just add the point again, this time on the correct curve.
            self.log(*args, **kwargs)


class VisdomTextLogger(BaseVisdomLogger):
    '''Creates a text window in visdom and logs output to it.

    The output can be formatted with fancy HTML, and it new output can
    be set to 'append' or 'replace' mode.

    Args:
        fields: Currently not used
        update_type: One of {'REPLACE', 'APPEND'}. Default 'REPLACE'.

    For examples, make sure that your visdom server is running.

    Example:
        >>> notes_logger = VisdomTextLogger(update_type='APPEND')
        >>> for i in range(10):
        >>>     notes_logger.log("Printing: {} of {}".format(i+1, 10))
        # results will be in Visdom environment (default: http://localhost:8097)

    '''
    valid_update_types = ['REPLACE', 'APPEND']

    def __init__(self, fields=None, win=None, env=None, opts={}, update_type=valid_update_types[0],
                 port=8097, server="localhost"):

        super(VisdomTextLogger, self).__init__(fields, win, env, opts, port, server)
        self.text = ''

        if update_type not in self.valid_update_types:
            raise ValueError("update type '{}' not found. Must be one of {}".format(
                update_type, self.valid_update_types))
        self.update_type = update_type

        self.viz_logger = self._viz_prototype(self.viz.text)

    def log(self, msg, *args, **kwargs):
        text = msg
        if self.update_type == 'APPEND' and self.text:
            self.text = "<br>".join([self.text, text])
        else:
            self.text = text
        self.viz_logger([self.text])

    def _log_all(self, stats, log_fields, prefix=None, suffix=None, require_dict=False):
        results = []
        for field_idx, field in enumerate(self.fields):
            parent, stat = None, stats
            for f in field:
                parent, stat = stat, stat[f]
            name, output = self._gather_outputs(field, log_fields,
                                                parent, stat, require_dict)
            if not output:
                continue
            self._align_output(field_idx, output)
            results.append((name, output))
        if not results:
            return
        output = self._join_results(results)
        if prefix is not None:
            self.log(prefix)
        self.log(output)
        if suffix is not None:
            self.log(suffix)

    def _align_output(self, field_idx, output):
        for output_idx, o in enumerate(output):
            if len(o) < self.field_widths[field_idx][output_idx]:
                num_spaces = self.field_widths[field_idx][output_idx] - len(o)
                output[output_idx] += ' ' * num_spaces
            else:
                self.field_widths[field_idx][output_idx] = len(o)

    def _join_results(self, results):
        joined_out = map(lambda i: (i[0], ' '.join(i[1])), results)
        joined_fields = map(lambda i: '{}: {}'.format(i[0], i[1]), joined_out)
        return '\t'.join(joined_fields)

    def _gather_outputs(self, field, log_fields, stat_parent, stat, require_dict=False):
        output = []
        name = ''
        if isinstance(stat, dict):
            log_fields = stat.get(log_fields, [])
            name = stat.get('log_name', '.'.join(field))
            for f in log_fields:
                output.append(f.format(**stat))
        elif not require_dict:
            name = '.'.join(field)
            number_format = stat_parent.get('log_format', '')
            unit = stat_parent.get('log_unit', '')
            fmt = '{' + number_format + '}' + unit
            output.append(fmt.format(stat))
        return name, output
