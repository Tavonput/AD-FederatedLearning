from ADFL.Driver.common import *

from ADFL.types import TrainingConfig


def test_get_delay_map():
    delay = TrainingConfig.Delay(
        compute_sigma=None,
        network_sigma=None,
        network_shift=0,
    )
    delay.delay_map = get_delay_map(delay, 10)
    print(delay.delay_map)

    delay = TrainingConfig.Delay(
        compute_sigma=1,
        network_sigma=1,
        network_shift=5,
    )
    delay.delay_map = get_delay_map(delay, 10)
    print(delay.delay_map)
