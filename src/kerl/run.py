from pytorch_lightning import seed_everything
from .system import KERLSystem
from .data import get_dataset

def run_system(args: dict):

    seed_everything(args["seed"])
    dataset = get_dataset(args["dataset"])(args, args["tokenizer"], args["restore"])
    # only has redial dataset for now
    system = KERLSystem(args, dataset)
    system.fit()


def check_restore(opt):
    """
    Check whether to restore the data or not.
    :param opt: the config dictionary
    :return: True or False
    """
    return bool(opt["restore"])
