import os
from absl import app
from absl import flags
from absl import logging
# from utils import GenerateTfrecord, HouseGan

here = os.path.dirname(os.path.abspath(__file__))

flags.DEFINE_enum(
    'mode',
    'eval_rollout',
    ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.'
)
flags.DEFINE_enum(
    'eval_split',
    'test',
    ['train', 'valid', 'test'],
    help='Split to use when running evaluation.'
)
flags.DEFINE_string(
    'data_path',
    'WaterDropSample',
    help='The dataset directory.'
)
flags.DEFINE_integer(
    'batch_size',
    1,
    help='The batch size.'
)
flags.DEFINE_integer(
    'num_steps',
    int(2e7),
    help='Number of steps of training.'
)
flags.DEFINE_float(
    'noise_std',
    6.7e-4,
    help='The std deviation of the noise.'
)
flags.DEFINE_string(
    'model_path',
    'models',
    help='The path for saving checkpoints of the model. Defaults to a temporary directory.'
)
flags.DEFINE_string(
    'output_path',
    'rollout/WaterDropSample/',
    help='The path for saving outputs (e.g. rollouts).'
)

FLAGS = flags.FLAGS

def main():
    ######################################
    if FLAGS.mode in ['train', 'eval_rollout']:
        learning = LearningSimulator(FLAGS.data_path, FLAGS.noise_std, FLAGS.noise_std)
        # if FLAGS.mode == 'train':
        #     learning.training()
        # else:
        #     learning.validation()

        learning.test_system()

    ################### 학습/평가 명령 ###################
    house_gan = HouseGan(FLAGS)
    house_gan.training()
    # house_gan.validation()
    # house_gan.test(os.path.join(here, HParams['test_csv']))

if __name__ == '__main__':
    main()