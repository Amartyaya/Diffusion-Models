import torch
import matplotlib.pyplot as plt


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
        img_dim = len(img_shape)

        for t in reversed(range(1, self.timesteps+1)):

            z = torch.randn_like(xt) if t > 1 else torch.zeros_like(xt)

            t_tensor = torch.tensor([t]).to(xt.device)

            beta_t  = self._extract(self.betas, t_tensor, img_dim)
            alpha_t = self._extract(self.alphas, t_tensor, img_dim)
            alpha_bar_t = self._extract(self.alpha_bars, t_tensor, img_dim)
            alpha_bar_t_minus_one = self._extract(self.alpha_bars, t_tensor-1, img_dim) 

            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

            sigma_t = torch.sqrt((1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t) * beta_t)

            xt = coeff1 * (xt - coeff2 * model(xt, t_tensor)) + z * sigma_t

        return xt

    @torch.inference_mode()
    def interpolate(self, data, idx1, idx2, model, t=256):

        img1 = data.dataset[idx1][0].unsqueeze(0).to(device=self.device, dtype=self.dtype)
        img2 = data.dataset[idx2][0].unsqueeze(0).to(device=self.device, dtype=self.dtype)

        t_tensor = torch.tensor([t]).to(self.device)
        img_dim = img1.ndim
        eps = torch.randn_like(img1)

        xt1 = self._extract(self.sqrt_alpha_bars, t_tensor, img_dim) * img1 + self._extract(self.sqrt_one_minus_alpha_bars, t_tensor, img_dim) * eps
        xt2 = self._extract(self.sqrt_alpha_bars, t_tensor, img_dim) * img2 + self._extract(self.sqrt_one_minus_alpha_bars, t_tensor, img_dim) * eps

        coeffs = torch.tensor([0.1 * i for i in range(11)]).to(device=self.device, dtype=self.dtype)[:, None, None, None]
        xt_interpolated = (1 - coeffs) * xt1 + coeffs * xt2

        for ti in reversed(range(1, t+1)):

            z = torch.randn_like(xt_interpolated) if ti > 1 else torch.zeros_like(xt_interpolated)
            ti_tensor = torch.full((len(coeffs),), ti).to(self.device)

            beta_t  = self._extract(self.betas, ti_tensor, img_dim)
            alpha_t = self._extract(self.alphas, ti_tensor, img_dim)
            alpha_bar_t = self._extract(self.alpha_bars, ti_tensor, img_dim)
            alpha_bar_t_minus_one = self._extract(self.alpha_bars, ti_tensor-1, img_dim)

            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

            sigma_t = torch.sqrt((1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t) * beta_t)

            xt_interpolated = coeff1 * (xt_interpolated - coeff2 * model(xt_interpolated, ti_tensor)) + z * sigma_t

        xt_list = list(torch.unbind(xt_interpolated))
        xt_list.insert(0, img1.squeeze(0))
        xt_list.append(img2.squeeze(0))

        _, axs = plt.subplots(nrows=1, ncols=len(coeffs)+2)
        vtransform = data.visual_transform
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(vtransform(xt_list[i]))
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()


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
