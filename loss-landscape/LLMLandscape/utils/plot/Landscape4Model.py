import os
import torch
from torch import nn
import numpy as np
import random
from typing import Callable, Iterable, Dict, List
from tqdm import tqdm
from matplotlib import pyplot as plt
from .ColorUtils import get_datetime_str
from utils.seed import set_seed


class Landscape4Model:
    def __init__(
        self,
        model: nn.Module,
        loss: Callable,
        mode: str = "2D",
        direction: str = "Gaussian",
        save_path: str = "./landscape/",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        :param model:
        :param loss: given a model, return loss
        """
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        assert mode in ["3D", "contour", "2D"]
        self.mode = mode
        assert direction in ["Gaussian", "WeightNormGaussian"]
        self.direction = direction
        self.model = model
        self.loss = loss
        self.device = device

    @torch.no_grad()
    def synthesize_coordinates(self, x_min=-0.02, x_max=0.02, x_interval=1e-3, y_min=-0.02, y_max=0.02, y_interval=1e-3):
        x = np.arange(x_min, x_max, x_interval)
        y = np.arange(y_min, y_max, y_interval)
        self.mesh_x, self.mesh_y = np.meshgrid(x, y)

    @torch.no_grad()
    def draw(self, random_direction: bool = True, saving_name: str = None, direction_seed: int | None = 42):
        if random_direction:
            self.find_direction(seed=direction_seed)
        z = self.compute_for_draw()
        self.draw_figure(self.mesh_x, self.mesh_y, z, saving_name=saving_name)
        return z

    @torch.no_grad()
    def find_direction(self, excluded_param_names: Iterable[str] = (), seed: int = None):
        self.x_unit_vector = {}
        self.y_unit_vector = {}
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        for name, param in self.model.named_parameters():
            if name in excluded_param_names:
                self.x_unit_vector[name] = self.y_unit_vector[name] = torch.tensor(0)
                continue
            x = torch.randn_like(param.data).to(self.device)

            self.x_unit_vector[name] = x if self.direction == "Gaussian" else x / x.norm() * param.norm()
            self.y_unit_vector[name] = torch.tensor(0)
            if self.mode in ["3D", "contour"]:
                y = torch.randn_like(param.data).to(self.device)
                self.y_unit_vector[name] = y if self.direction == "Gaussian" else y / y.norm() * param.norm()

    @torch.no_grad()
    def compute_for_draw(self):
        result = []
        if self.mode == "2D":
            mesh_x = self.mesh_x[0]
            for i in tqdm(range(mesh_x.shape[0])):
                now_x = mesh_x[i]
                loss = self._compute_loss_for_one_coordinate(now_x, 0)
                result.append(loss)
        else:
            mesh_x = self.mesh_x
            for i in tqdm(range(mesh_x.shape[0])):
                for j in range(mesh_x.shape[1]):
                    now_x = mesh_x[i, j]
                    now_y = self.mesh_y[i, j]
                    loss = self._compute_loss_for_one_coordinate(now_x, now_y)
                    result.append(loss)

        result = np.array(result)
        result = result.reshape(mesh_x.shape)
        return result

    def add_perturbation_to_params(self, now_x: float, now_y: float):
        for name, param in self.model.named_parameters():
            param.add_(now_x * self.x_unit_vector[name].to(param.device))
            param.add_(now_y * self.y_unit_vector[name].to(param.device))

    @torch.no_grad()
    def _recover_params(self, now_x: float, now_y: float):
        self.add_perturbation_to_params(-now_x, -now_y)

    @torch.no_grad()
    def _compute_loss_for_one_coordinate(self, now_x: float, now_y: float):
        self.add_perturbation_to_params(now_x, now_y)
        loss = self.loss(self.model)
        self._recover_params(now_x, now_y)
        return loss

    def draw_figure(self, mesh_x, mesh_y, mesh_z, saving_name: str = None):
        if self.mode == "3D":
            figure = plt.figure()
            axes = figure.add_subplot(111, projection="3d")
            axes.plot_surface(mesh_x, mesh_y, mesh_z, cmap="rainbow")
        elif self.mode == "2D":
            plt.plot(mesh_x[0], mesh_z)
        plt.savefig(os.path.join(self.save_path, saving_name or get_datetime_str() + ".png"))
        plt.close()


class Landscape4ModelPCA(Landscape4Model):
    """Landscape drawer using top PCA/SVD components instead of random directions.

    For each parameter tensor, we compute rank-1 and rank-2 components (when available)
    via low-rank SVD on a flattened 2D view. These become x/y perturbation directions.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable,
        mode: str = "2D",
        save_path: str = "./landscape/",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_elements_for_pca: int = 20_000_000,
        k_components: int = 1,
    ):
        super().__init__(model=model, loss=loss, mode=mode, direction="WeightNormGaussian", save_path=save_path, device=device)
        self.max_elements_for_pca = max_elements_for_pca
        self.k_components = k_components
        self._pca_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def _cached_svd(self, param: torch.Tensor, param_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._pca_cache.get(param_name)
        if cached is not None:
            return cached

        matrix = param.detach().float()
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        self._pca_cache[param_name] = (U, S, Vh)
        return U, S, Vh

    @torch.no_grad()
    def warmup_cache(self, excluded_param_names: Iterable[str] = ()) -> None:
        """Precompute and cache SVD bases for all parameters once.

        This removes the repeated SVD cost when sweeping k_components.
        """
        excluded = set(excluded_param_names)
        for name, param in self.model.named_parameters():
            if name in excluded:
                continue
            if param.ndim != 2:
                continue
            if param.numel() > self.max_elements_for_pca:
                continue
            self._cached_svd(param.data, name)

    @torch.no_grad()
    def _rank_k_directions(self, param: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        if param.ndim == 1:
            x_dir = param.detach().float().reshape(-1)
            y_dir = torch.zeros_like(x_dir)
            return x_dir.reshape_as(param).to(param.dtype), y_dir.reshape_as(param).to(param.dtype)

        if param.ndim == 2:
            raise RuntimeError("_rank_k_directions requires param_name after cache refactor")

        raise ValueError(f"Unsupported parameter shape: {param.shape}")

    @torch.no_grad()
    def _rank_k_directions_for_name(self, param_name: str, param: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if param.ndim == 1:
            x_dir = param.detach().float().reshape(-1)
            y_dir = torch.zeros_like(x_dir)
            return x_dir.reshape_as(param).to(param.dtype), y_dir.reshape_as(param).to(param.dtype)

        if param.ndim == 2:
            U, S, Vh = self._cached_svd(param, param_name)
            # k_components is 1-based component index: 1->PC1, 2->PC2, ...
            comp_idx = max(0, min(self.k_components - 1, S.numel() - 1))

            # Single principal component direction only (no singular-value weighting).
            x_rank = torch.outer(U[:, comp_idx], Vh[comp_idx, :])

            # Keep y as zero for now (2D-only PCA path).
            y_rank = torch.zeros_like(x_rank)

            return x_rank.to(param.dtype), y_rank.to(param.dtype)

        raise ValueError(f"Unsupported parameter shape: {param.shape}")

    @torch.no_grad()
    def find_direction(self, excluded_param_names: Iterable[str] = (), seed: int = None):
        self.x_unit_vector = {}
        self.y_unit_vector = {}

        for name, param in self.model.named_parameters():
            if name in excluded_param_names:
                self.x_unit_vector[name] = self.y_unit_vector[name] = torch.tensor(0, device=param.device, dtype=param.dtype)
                continue

            x_dir, y_dir = self._rank_k_directions_for_name(name, param.data)
            self.x_unit_vector[name] = x_dir.to(param.device)
            self.y_unit_vector[name] = y_dir.to(param.device)


def calculate_whole_loss(
    input_model: nn.Module,
    loader: Iterable,
    criterion: Callable = nn.CrossEntropyLoss(),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> float:
    input_model.eval()
    total_loss, denom = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        cur_loss = criterion(input_model(x), y)
        total_loss += cur_loss.item() * x.shape[0]
        denom += x.shape[0]
    total_loss /= denom
    return total_loss


def draw_classifier_loss_landscape(
    model: nn.Module,
    loader: Iterable,
    mode: str = "2D",
    criterion: Callable = nn.CrossEntropyLoss(),
    save_path: str = "./landscape/",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    coordinate_config: Dict = None,
):
    coordinate_config = dict(
        x_min=-0.02, x_max=0.02, x_interval=1e-3, y_min=-0.02, y_max=0.02, y_interval=1e-3, **coordinate_config
    )

    drawer = Landscape4Model(
        model,
        lambda m: calculate_whole_loss(m, loader, criterion, device),
        mode,
        save_path,
        device,
    )
    drawer.synthesize_coordinates(**coordinate_config)
    drawer.draw()


def draw_classifier_loss_landscape_each_class(
    model: nn.Module,
    loader: Iterable,
    mode: str = "2D",
    criterion: Callable = nn.CrossEntropyLoss(),
    save_path: str = "./landscape/",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    coordinate_config: Dict = None,
):
    raise NotImplemented
    coordinate_config = dict(
        x_min=-0.02, x_max=0.02, x_interval=1e-3, y_min=-0.02, y_max=0.02, y_interval=1e-3, **coordinate_config
    )
    xs, ys = zip(*loader)
    xs, ys = list(xs), torch.cat(list(ys), dim=0)
    seed = random.randint(1, 10000)
    drawer = Landscape4Model(
        model,
        lambda m: calculate_whole_loss(m, loader, criterion, device),
        mode,
        save_path,
        device,
    )
    drawer.synthesize_coordinates(**coordinate_config)
    for y in range(ys.max().item()):
        set_seed(seed)
        cur_loader = [(x, torch.zeros(x.shape[0]) + y) for x in xs]
        drawer.loss = lambda m: calculate_whole_loss(m, cur_loader, criterion, device)
        drawer.draw()
