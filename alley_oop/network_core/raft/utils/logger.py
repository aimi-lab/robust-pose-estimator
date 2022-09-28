import wandb

class Logger:
    def __init__(self, model, scheduler, config, project_name, log):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.log = log
        if log:
            wandb.init(project=project_name, config=config)
        self.header = False

    def _print_header(self):
        metrics_data = [k for k in sorted(self.running_loss.keys())]
        training_str = "[steps, lr] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:<15}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

    def _print_training_status(self):
        if not self.header:
            self.header = True
            self._print_header()
        metrics_data = [self.running_loss[k] for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        for k in self.running_loss:
            self.running_loss[k] = 0.0

    def push(self, metrics, freq):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key] / freq

    def write_dict(self, results):
        wandb.log(results)

    def flush(self):
        if self.log:
            self.write_dict(self.running_loss)
        self._print_training_status()
        self.running_loss = {}

    def close(self):
        wandb.finish()

    def save_model(self, path):
        if self.log:
            wandb.save(path)

    def log_plot(self, fig):
        if self.log:
            wandb.log({"optical flow": fig})