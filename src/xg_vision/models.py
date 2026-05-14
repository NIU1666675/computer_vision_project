from __future__ import annotations

import warnings
from typing import Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBackbone(nn.Module):
    """Small CNN trained from scratch for the visual xG baseline."""

    def __init__(self, channels: Sequence[int] = (32, 64, 128, 256), feature_dim: int = 256) -> None:
        super().__init__()
        blocks = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(ConvBlock(in_channels, int(out_channels)))
            in_channels = int(out_channels)
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.projection(x)


class TorchVisionBackbone(nn.Module):
    """ImageNet backbone used when torchvision is available."""

    def __init__(self, name: str = "resnet18", feature_dim: int = 256, pretrained: bool = True, freeze: bool = True) -> None:
        super().__init__()
        try:
            import torchvision.models as tv_models
        except ImportError as exc:
            raise ImportError("Install torchvision to use pretrained backbones.") from exc

        name = name.lower()
        if name != "resnet18":
            raise ValueError("Only resnet18 is currently supported as pretrained backbone")

        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
        in_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        self.encoder = model
        self.projection = nn.Sequential(
            nn.Linear(in_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim
        self.frozen = freeze
        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)
        return self.projection(x)


def make_backbone(
    backbone: str = "custom",
    cnn_channels: Sequence[int] = (32, 64, 128, 256),
    feature_dim: int = 256,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    name = backbone.lower()
    if name == "auto":
        try:
            module = TorchVisionBackbone("resnet18", feature_dim, pretrained, freeze_backbone)
            module.resolved_backbone = "resnet18"  # type: ignore[attr-defined]
            return module
        except Exception as exc:
            warnings.warn(f"Falling back to custom CNN backbone because resnet18 could not be loaded: {exc}")
            module = ConvBackbone(cnn_channels, feature_dim)
            module.resolved_backbone = "custom"  # type: ignore[attr-defined]
            return module
    if name in {"custom", "small_cnn", "cnn"}:
        module = ConvBackbone(cnn_channels, feature_dim)
        module.resolved_backbone = "custom"  # type: ignore[attr-defined]
        return module
    if name == "resnet18":
        module = TorchVisionBackbone(name, feature_dim, pretrained, freeze_backbone)
        module.resolved_backbone = "resnet18"  # type: ignore[attr-defined]
        return module
    raise ValueError("backbone must be one of: auto, custom, resnet18")


class BinaryHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class CNNClassifier(nn.Module):
    def __init__(
        self,
        cnn_channels: Sequence[int] = (32, 64, 128, 256),
        feature_dim: int = 256,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.25,
        backbone: str = "custom",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = make_backbone(backbone, cnn_channels, feature_dim, pretrained, freeze_backbone)
        self.resolved_backbone = getattr(self.backbone, "resolved_backbone", backbone)
        self.classifier = BinaryHead(feature_dim, classifier_hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


class CNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        cnn_channels: Sequence[int] = (32, 64, 128, 256),
        feature_dim: int = 256,
        classifier_hidden_dim: int = 128,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.25,
        backbone: str = "custom",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = make_backbone(backbone, cnn_channels, feature_dim, pretrained, freeze_backbone)
        self.resolved_backbone = getattr(self.backbone, "resolved_backbone", backbone)
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = BinaryHead(lstm_hidden_dim, classifier_hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        x = x.reshape(batch * steps, channels, height, width)
        features = self.backbone(x).reshape(batch, steps, -1)
        _, (hidden, _) = self.lstm(features)
        return self.classifier(hidden[-1])


class CNNAttentionClassifier(nn.Module):
    def __init__(
        self,
        cnn_channels: Sequence[int] = (32, 64, 128, 256),
        feature_dim: int = 256,
        classifier_hidden_dim: int = 128,
        sequence_length: int = 6,
        attention_heads: int = 4,
        attention_layers: int = 2,
        attention_ff_dim: int = 512,
        dropout: float = 0.25,
        backbone: str = "custom",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = make_backbone(backbone, cnn_channels, feature_dim, pretrained, freeze_backbone)
        self.resolved_backbone = getattr(self.backbone, "resolved_backbone", backbone)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(1, sequence_length + 1, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=attention_heads,
            dim_feedforward=attention_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=attention_layers)
        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = BinaryHead(feature_dim, classifier_hidden_dim, dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        x = x.reshape(batch * steps, channels, height, width)
        features = self.backbone(x).reshape(batch, steps, -1)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, features], dim=1)
        tokens = tokens + self.positional_embedding[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        return self.classifier(self.norm(encoded[:, 0]))


def canonical_model_name(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    aliases = {
        "cnn": "cnn",
        "baseline": "cnn",
        "lstm": "lstm",
        "cnn_lstm": "lstm",
        "attention": "attention",
        "attn": "attention",
        "cnn_attention": "attention",
    }
    if normalized not in aliases:
        raise ValueError("model must be one of: cnn, lstm, attention")
    return aliases[normalized]


def model_input_type(name: str) -> str:
    return "single" if canonical_model_name(name) == "cnn" else "sequence"


def model_kwargs_from_config(config: dict, model_name: str) -> dict:
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    common = {
        "cnn_channels": tuple(model_cfg.get("cnn_channels", (32, 64, 128, 256))),
        "feature_dim": int(model_cfg.get("feature_dim", 256)),
        "classifier_hidden_dim": int(model_cfg.get("classifier_hidden_dim", 128)),
        "dropout": float(train_cfg.get("dropout", 0.25)),
        "backbone": str(model_cfg.get("backbone", "custom")),
        "pretrained": bool(model_cfg.get("pretrained", True)),
        "freeze_backbone": bool(model_cfg.get("freeze_backbone", True)),
    }
    name = canonical_model_name(model_name)
    if name == "cnn":
        return common
    if name == "lstm":
        return {
            **common,
            "lstm_hidden_dim": int(model_cfg.get("lstm_hidden_dim", 256)),
            "lstm_layers": int(model_cfg.get("lstm_layers", 1)),
        }
    return {
        **common,
        "sequence_length": int(data_cfg.get("sequence_length", 6)),
        "attention_heads": int(model_cfg.get("attention_heads", 4)),
        "attention_layers": int(model_cfg.get("attention_layers", 2)),
        "attention_ff_dim": int(model_cfg.get("attention_ff_dim", 512)),
    }


def build_model(model_name: str, **kwargs) -> nn.Module:
    name = canonical_model_name(model_name)
    if name == "cnn":
        return CNNClassifier(**kwargs)
    if name == "lstm":
        return CNNLSTMClassifier(**kwargs)
    if name == "attention":
        return CNNAttentionClassifier(**kwargs)
    raise AssertionError("unreachable")
