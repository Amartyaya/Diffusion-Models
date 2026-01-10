import torch


class DDPM:

    def __init__(self, beta_start, beta_end, timesteps, device="cpu", dtype=torch.float32):

        self.dtype = dtype
        self.device = device
        self.timesteps = timesteps

        self.betas = torch.linspace(beta_start, beta_end, self.timesteps, device=self.device, dtype=self.dtype)
        # The 0th index is treated as giving no noise, i.e. it is the original image.
        self.betas = torch.cat((torch.tensor([0.]).to(self.betas), self.betas), dim=-1)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)

    @staticmethod
    def _extract(vals, t, x_dim):
        """
        Extract the values associated with the given times and view them as [batch_size, 1, 1, ...].

        Args:
            vals: (timesteps,), mostly represents variance schedule and similar values
            t: (batch_size,), represents sampled time values for an input batch
            x_dim: represents the number of dimensions of the input batch
        """
        return vals.gather(-1, t).view(t.shape[0], *((1,) * (x_dim - 1)))

    def forward_sample(self, x0, t):
        """Return xt ~ q(xt|x0) and the corresponding normal noise."""

        assert x0.shape[0] == t.shape[0]
        eps = torch.randn_like(x0)
        return (
            self._extract(self.sqrt_alpha_bars, t, x0.ndim) * x0 + \
            self._extract(self.sqrt_one_minus_alpha_bars, t, x0.ndim) * eps,
            eps
        )

    @torch.inference_mode()
    def generate(self, model, img_shape):
        """Implement Algorithm-2 given in the DDPM paper."""

        xt = torch.randn(*img_shape).to(self.device)

        for t in reversed(range(1, self.timesteps+1)):

            z = torch.randn_like(xt) if t > 1 else torch.zeros_like(xt)

            t_tensor = torch.tensor([t]).to(xt.device)

            beta_t  = self._extract(self.betas, t_tensor, len(img_shape))
            alpha_t = self._extract(self.alphas, t_tensor, len(img_shape))
            alpha_bar_t = self._extract(self.alpha_bars, t_tensor ,len(img_shape))
            alpha_bar_t_minus_one = self._extract(self.alpha_bars, t_tensor-1 ,len(img_shape))

            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

            sigma_t = torch.sqrt((1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t) * beta_t)

            xt = coeff1 * (xt - coeff2 * model(xt, t_tensor)) + z * sigma_t

        return xt

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ddpm = DDPM(1e-4, 2e-2, 10, device)
    x0 = torch.ones((5, 2, 2))
    t = torch.randint(0, 11, (5,))
    print(t)
    print(ddpm.forward_sample(x0, t))

    return



if __name__ == "__main__":
    main()
