from modules.utils.util import generate_out_folder


class BasicModel:
    def init():
        pass

    def set_args(self, args):
        self.args = args
        self.out_folder = generate_out_folder(
            self.args.output,
            self.args.training_data,
            self.args.dataset_division,
            self.__class__.__name__,
        )

    def __init__(self) -> None:
        pass


class STDGISequential(BasicModel):
    def __init__(self):
        pass

    def get_data_loader(self):
        pass

    def init():
        pass

    def train_stdgi(self):
        pass

    def train_decoder(self):
        pass

    def training(self):
        pass

    def test(self):
        pass

    def save_logs(self):
        pass

    def run(self):
        pass
