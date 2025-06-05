from ADFL.model import get_mobile_net_v3_small
import torch.nn as nn

def main():
    model = get_mobile_net_v3_small(10)
    old_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1,
        old_conv.out_channels,
        old_conv.kernel_size,
        old_conv.stride,
        old_conv.padding,
        bias=False,
    )
    print(old_conv)
    print(model)
    return

if __name__ == "__main__":
    main()
