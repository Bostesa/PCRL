"""Small regressions for redesign selection, calibration and checkpoint semantics."""
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from pcrl.data.base import collate_pcrl_batch
from pcrl.models.encoder import StandardEncoder
from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec
from pcrl.training.independence.vclub import VCLUB
from pcrl.training.v2_trainer import V2EpochMetrics, V2Trainer, V2TrainerConfig


def make_trainer(tmp_path, *, classes=2, per_class_threshold=6, erase_mode=None):
    torch.manual_seed(91)
    registry = PurposeRegistry()
    for name, attr in [("p1", "V"), ("p2", "U")]:
        registry.register(PurposeSpec(name=name, allowed_tasks=[name], disallowed_attrs=[attr],
                                     allowed_task_dims={name: 2}, disallowed_attr_dims={attr: classes}))
    backbone = StandardEncoder(3, [3], 3, dropout=0.7,
                               use_erase_layer=erase_mode is not None,
                               erase_mode=erase_mode or "shared_union", n_purposes=2)
    encoder = PerPurposeLoRAEncoder(backbone, 2, rank=2, dropout=0.2,
                                    lora_target="repr_proj_only")
    vclubs = {f"{p.name}__{p.disallowed_attrs[0]}": VCLUB(3, classes, hidden_dim=4, z_categorical=True)
              for p in registry.purposes}
    config = V2TrainerConfig(epochs=1, warmup_epochs=0, leace_init=False, lambda_vicreg=0,
                             vclub_steps=1, checkpoint_dir=str(tmp_path),
                             cross_purpose_attrs=["U"], cross_purpose_threshold=0.1,
                             per_class_constraint_threshold=per_class_threshold,
                             erase_mode=erase_mode or "shared_union")
    return V2Trainer(encoder, {p.name: nn.Linear(3, 2) for p in registry.purposes},
                     vclubs, registry, config)


def loader(batch_size=8, classes=2):
    rows = []
    for i in range(16):
        u, v = (i // 2) % 2, i % 2
        rows.append({"features": torch.tensor([2*u-1., 2*v-1., float(2*((i // 4) % 2)-1)]),
                     "task_labels": {"p1": torch.tensor(u), "p2": torch.tensor(v)},
                     "sensitive_attrs": {"U": torch.tensor(u), "V": torch.tensor(v)}})
    return DataLoader(rows, batch_size=batch_size, collate_fn=collate_pcrl_batch)


def test_selection_uses_per_constraint_thresholds_and_recomputes_violation(tmp_path):
    trainer = make_trainer(tmp_path)
    feasible = {"p1__V": .04, "p2__U": .04, "concat[U]": .08}
    violated = {"p1__V": .06, "p2__U": .04, "concat[U]": 0.0}
    selected = trainer._select_cotter_best([
        (1, {}, 1.0, feasible, 999., False),
        (2, {}, .95, violated, 0., False),
    ], .05)
    assert selected[0] == 1 and selected[-1] == "feasible"
    assert selected[4] == 0
    assert trainer._score_summary({**feasible, "concat[U]": .1})[0]
    # Both infeasible: the cross-purpose excess is .01, not .06.
    cross = {**feasible, "concat[U]": .11}
    per = {**feasible, "p1__V": .07}
    selected = trainer._select_cotter_best([(1, {}, 1., cross, 999., False),
                                            (2, {}, 1., per, 0., False)])
    assert selected[0] == 1 and selected[-1] == "fallback"


def test_missing_or_undefined_validation_scores_are_not_feasible(tmp_path):
    trainer = make_trainer(tmp_path)
    for scores in [{}, {"p1__V": 0.}, {"p1__V": 0., "p2__U": 0., "concat[U]": float("nan")}]:
        assert trainer._score_summary(scores) == (False, float("inf"))


def test_initialization_checkpoint_precedes_every_optimizer_update(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    trainer.config.epochs = 2
    # Isolate checkpoint metadata: validation selects the first trained epoch.
    def validation(_loader):
        return V2EpochMetrics(task_loss=float(trainer.state.epoch),
                              r2_per_pair={k: .0 for k in trainer.score_thresholds})
    monkeypatch.setattr(trainer, "evaluate", validation)
    initial = {k: v.clone() for k, v in trainer.encoder.state_dict().items()}
    trainer.train(loader(), loader())
    init = torch.load(tmp_path / "initialization.pt", weights_only=False)
    final = torch.load(tmp_path / "final.pt", weights_only=False)
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert init["state"]["epoch"] == 0
    assert init["optimizer_steps"] == {"primal": 0, "vclub": 0}
    for k, value in init["lora_adapters"].items():
        assert torch.equal(value, initial[f"adapters.{k}"])
    assert final["state"]["epoch"] == 2
    assert final["optimizer_steps"] == {"primal": 4, "vclub": 4}
    assert best["state"]["epoch"] == 1
    assert best["state"]["global_step"] == 2
    assert best["optimizer_steps"] == {"primal": 2, "vclub": 2}
    assert any(not torch.equal(best["vclubs"][k], value)
               for k, value in final["vclubs"].items())


def test_frozen_backbone_bn_and_dropout_modes_match_calibration(tmp_path):
    trainer = make_trainer(tmp_path)
    encoder = trainer.encoder
    encoder.train()
    assert encoder.training and encoder.adapters.training
    assert all(not module.training for module in encoder.backbone.modules())
    x = torch.randn(12, 3)
    before = encoder.backbone(x)
    encoder.eval()
    after = encoder.backbone(x)
    assert torch.equal(before, after)
    encoder.train()
    assert encoder.adapters[0][0].dropout.training


def test_validation_full_split_coverage_and_absent_dual_skips(tmp_path):
    trainer = make_trainer(tmp_path, classes=3, per_class_threshold=3)
    stats = trainer.evaluate(loader())
    assert math.isnan(stats.r2_per_pair["p1__V"])
    assert stats.class_coverage["p1__V"]["support"] == [8, 8, 0]
    assert stats.class_coverage["p1__V"]["valid_mask"] == [True, True, False]
    constraint = trainer.proxy.constraints["p1__V__class_2"]
    before = constraint.lambda_value
    trainer._primal_and_dual_step(next(iter(loader())))
    assert constraint.lambda_value == before
    assert not constraint.value_history


def test_per_purpose_fitted_erasure_preserves_own_signal(tmp_path):
    outputs = {}
    for mode in ["shared_union", "per_purpose"]:
        trainer = make_trainer(tmp_path / mode, erase_mode=mode)
        backbone = trainer.encoder.backbone
        backbone.network = nn.Identity()
        with torch.no_grad():
            backbone.repr_proj.weight.copy_(torch.eye(3))
            backbone.repr_proj.bias.zero_()
        trainer.fit_erase_layer(loader())
        trainer.encoder.eval()
        data = next(iter(loader(batch_size=16)))
        outputs[mode] = [trainer.encoder(data["features"], p) for p in range(2)]
    # h[0] is U and h[1] is V. Per-purpose maps erase the conflicting
    # purpose's task while retaining their own task; union erases both.
    for p, own_axis in [(0, 0), (1, 1)]:
        assert outputs["per_purpose"][p][:, own_axis].std() > .9
        assert outputs["per_purpose"][p][:, 1-own_axis].std() < 1e-5
        assert outputs["shared_union"][p][:, :2].std(0).max() < 1e-5
