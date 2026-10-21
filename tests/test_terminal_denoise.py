import torch

from compass.Sampler import Sampler


class _ConstantLambdaSDE:
    def lambda_t(self, t):
        return torch.full_like(t, 0.5)


class _SBIm:
    sde = _ConstantLambdaSDE()


def test_terminal_denoise_uses_tweedie_increment_and_preserves_conditions():
    sampler = Sampler(_SBIm())
    sampler.timesteps_list = torch.tensor([0.001])
    sampler.cfg_alpha = None
    sampler._get_score = lambda data, t, mask, cfg: -2.0 * data

    data = torch.tensor([[[2.0, 7.0], [-4.0, 9.0]]])
    mask = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])

    denoised = sampler._terminal_denoise(data, mask)

    torch.testing.assert_close(
        denoised,
        torch.tensor([[[1.0, 7.0], [-2.0, 9.0]]]),
    )
