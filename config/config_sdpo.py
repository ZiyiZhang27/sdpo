import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base_sdpo.py"))


def aesthetic():
    config = base.get_config()

    config.pretrained.model = "stabilityai/sd-turbo"

    # number of sampler inference steps.
    config.sample.num_steps = 50

    config.num_epochs = 100

    config.reward_fn = "aesthetic_score"
    config.reward_name = "aesthetic"

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 8

    config.prompt_fn = "simple_animals"
    config.per_step_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }

    return config


def imagereward():
    config = aesthetic()

    config.num_epochs = 200

    config.reward_fn = "ImageReward"
    config.reward_name = config.reward_fn

    config.train.learning_rate = 5e-5
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 32

    return config


def hpsv2():
    config = aesthetic()

    config.num_epochs = 200

    config.reward_fn = "hpsv2"
    config.reward_name = config.reward_fn

    config.train.learning_rate = 5e-5
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 32

    return config


def pickscore():
    config = aesthetic()

    config.num_epochs = 200

    config.reward_fn = "PickScore"
    config.reward_name = config.reward_fn

    config.train.learning_rate = 6e-5
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 16

    return config


def get_config(name):
    return globals()[name]()
