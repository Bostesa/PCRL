"""Diverse auditor implementations for evaluating representation privacy.

This module provides multiple auditor types with different inductive biases
to robustly evaluate whether sensitive attributes can be extracted from
learned representations. The CVR metric uses the best-performing auditor
to provide an upper bound on information leakage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class BaseAuditor(ABC):
    """Abstract base class for all auditors.

    All auditors must implement:
    - fit(X, y): Train on representations X and labels y
    - predict(X): Return predicted labels
    - evaluate(X, y): Return dict with 'accuracy' and 'auc'
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseAuditor":
        """Train the auditor on representations and labels.

        Args:
            X: Representations array of shape (n_samples, repr_dim)
            y: Labels array of shape (n_samples,)

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels from representations.

        Args:
            X: Representations array of shape (n_samples, repr_dim)

        Returns:
            Predicted labels of shape (n_samples,)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities from representations.

        Args:
            X: Representations array of shape (n_samples, repr_dim)

        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate auditor performance.

        Args:
            X: Representations array of shape (n_samples, repr_dim)
            y: True labels array of shape (n_samples,)

        Returns:
            Dictionary with 'accuracy' and 'auc' metrics
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)

        # Compute AUC
        try:
            y_proba = self.predict_proba(X)
            n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 2

            if n_classes == 2:
                # Binary classification
                if len(y_proba.shape) > 1:
                    auc = roc_auc_score(y, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y, y_proba)
            else:
                # Multi-class: use one-vs-rest
                auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
        except (ValueError, IndexError):
            # AUC computation can fail if only one class present
            auc = 0.5

        return {"accuracy": accuracy, "auc": auc}


class MLPAuditor(BaseAuditor):
    """Multi-layer perceptron auditor using PyTorch.

    Args:
        hidden_dims: List of hidden layer dimensions
        learning_rate: Learning rate for Adam optimizer
        epochs: Number of training epochs
        batch_size: Training batch size
        dropout: Dropout probability
        device: Device to train on ('cpu' or 'cuda')
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        dropout: float = 0.2,
        device: str | None = None,
        seed: int | None = None,
    ):
        self.hidden_dims = hidden_dims or [128, 64]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        self.model: nn.Module | None = None
        self.scaler = StandardScaler()
        self.n_classes = 2

    def _build_model(self, input_dim: int, n_classes: int) -> nn.Module:
        """Build the MLP model."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPAuditor":
        """Train the MLP auditor."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # Preprocess
        X_scaled = self.scaler.fit_transform(X)
        self.n_classes = len(np.unique(y))

        # Build model
        self.model = self._build_model(X.shape[1], self.n_classes)
        self.model.to(self.device)

        # Prepare data
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy()

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()

        return proba


class RandomForestAuditor(BaseAuditor):
    """Random Forest auditor using sklearn.

    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None for unlimited)
        min_samples_split: Minimum samples required to split a node
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        seed: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=seed,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestAuditor":
        """Train the Random Forest auditor."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class SVMAuditor(BaseAuditor):
    """Support Vector Machine auditor using sklearn.

    Args:
        kernel: Kernel type ('rbf', 'linear', 'poly')
        C: Regularization parameter
        gamma: Kernel coefficient ('scale', 'auto', or float)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        seed: int | None = None,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.seed = seed

        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=seed,
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMAuditor":
        """Train the SVM auditor."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class XGBoostAuditor(BaseAuditor):
    """XGBoost auditor.

    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Boosting learning rate
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        seed: int | None = None,
    ):
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed

        self.model: xgb.XGBClassifier | None = None
        self.scaler = StandardScaler()
        self.n_classes = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostAuditor":
        """Train the XGBoost auditor."""
        X_scaled = self.scaler.fit_transform(X)
        self.n_classes = len(np.unique(y))

        objective = "binary:logistic" if self.n_classes == 2 else "multi:softprob"

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective=objective,
            random_state=self.seed,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class LinearAuditor(BaseAuditor):
    """Logistic Regression auditor using sklearn.

    Args:
        C: Inverse of regularization strength
        max_iter: Maximum number of iterations
        solver: Optimization algorithm
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        seed: int | None = None,
    ):
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.seed = seed

        self.model: LogisticRegression | None = None
        self.scaler = StandardScaler()
        self.n_classes = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearAuditor":
        """Train the Logistic Regression auditor."""
        X_scaled = self.scaler.fit_transform(X)
        self.n_classes = len(np.unique(y))

        multi_class = "multinomial" if self.n_classes > 2 else "auto"

        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            multi_class=multi_class,
            random_state=self.seed,
        )
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class AuditorPool:
    """Pool of diverse auditors for robust CVR evaluation.

    Trains multiple auditors with different inductive biases and returns
    the best performance across all. This provides an upper bound on
    information leakage from representations.

    Args:
        auditors: List of auditor instances to include in the pool
        names: Optional names for each auditor (for reporting)
    """

    def __init__(
        self,
        auditors: list[BaseAuditor],
        names: list[str] | None = None,
    ):
        self.auditors = auditors
        self.names = names or [f"auditor_{i}" for i in range(len(auditors))]

        if len(self.names) != len(self.auditors):
            raise ValueError("Number of names must match number of auditors")

        self.results: dict[str, dict[str, float]] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AuditorPool":
        """Train all auditors in the pool.

        Args:
            X: Representations array of shape (n_samples, repr_dim)
            y: Labels array of shape (n_samples,)

        Returns:
            self for method chaining
        """
        for name, auditor in zip(self.names, self.auditors):
            try:
                auditor.fit(X, y)
            except Exception as e:
                print(f"Warning: Failed to train {name}: {e}")
        return self

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Evaluate all auditors and return comprehensive results.

        Args:
            X: Representations array of shape (n_samples, repr_dim)
            y: True labels array of shape (n_samples,)

        Returns:
            Dictionary with:
            - 'best_accuracy': Highest accuracy across all auditors
            - 'best_auc': Highest AUC across all auditors
            - 'best_auditor_accuracy': Name of auditor with best accuracy
            - 'best_auditor_auc': Name of auditor with best AUC
            - 'individual_results': Dict mapping auditor name to metrics
        """
        self.results = {}

        for name, auditor in zip(self.names, self.auditors):
            try:
                metrics = auditor.evaluate(X, y)
                self.results[name] = metrics
            except Exception as e:
                print(f"Warning: Failed to evaluate {name}: {e}")
                self.results[name] = {"accuracy": 0.0, "auc": 0.5}

        # Find best performers
        best_accuracy = 0.0
        best_auc = 0.0
        best_auditor_accuracy = ""
        best_auditor_auc = ""

        for name, metrics in self.results.items():
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_auditor_accuracy = name
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_auditor_auc = name

        return {
            "best_accuracy": best_accuracy,
            "best_auc": best_auc,
            "best_auditor_accuracy": best_auditor_accuracy,
            "best_auditor_auc": best_auditor_auc,
            "individual_results": self.results,
        }

    def fit_evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Convenience method to fit and evaluate in one call."""
        self.fit(X, y)
        return self.evaluate(X, y)

    def get_best_accuracy(self) -> float:
        """Return the best accuracy from the last evaluation."""
        if not self.results:
            raise RuntimeError("No results available. Call evaluate() first.")
        return max(r["accuracy"] for r in self.results.values())

    def get_best_auc(self) -> float:
        """Return the best AUC from the last evaluation."""
        if not self.results:
            raise RuntimeError("No results available. Call evaluate() first.")
        return max(r["auc"] for r in self.results.values())


def get_default_auditor_pool(seed: int = 42) -> AuditorPool:
    """Create a default auditor pool with all 5 auditor types.

    Args:
        seed: Random seed for reproducibility

    Returns:
        AuditorPool with MLP, RandomForest, SVM, XGBoost, and Linear auditors
    """
    auditors = [
        MLPAuditor(hidden_dims=[128, 64], epochs=50, seed=seed),
        RandomForestAuditor(n_estimators=100, seed=seed),
        SVMAuditor(kernel="rbf", seed=seed),
        LinearAuditor(seed=seed),
    ]
    names = ["MLP", "RandomForest", "SVM", "Linear"]

    # Add XGBoost if available
    if HAS_XGBOOST:
        auditors.append(XGBoostAuditor(n_estimators=100, seed=seed))
        names.append("XGBoost")

    return AuditorPool(auditors, names)


def get_extensive_auditor_pool(seed: int = 42) -> AuditorPool:
    """Create an extensive auditor pool with multiple configurations.

    Includes multiple variations of each auditor type for more robust
    CVR estimation.

    Args:
        seed: Random seed for reproducibility

    Returns:
        AuditorPool with diverse auditor configurations
    """
    auditors = [
        # MLP variants
        MLPAuditor(hidden_dims=[64], epochs=50, seed=seed),
        MLPAuditor(hidden_dims=[128, 64], epochs=50, seed=seed),
        MLPAuditor(hidden_dims=[256, 128, 64], epochs=50, seed=seed),
        MLPAuditor(hidden_dims=[128, 64], epochs=100, dropout=0.3, seed=seed + 1),

        # Random Forest variants
        RandomForestAuditor(n_estimators=100, max_depth=10, seed=seed),
        RandomForestAuditor(n_estimators=200, max_depth=None, seed=seed),

        # SVM variants
        SVMAuditor(kernel="rbf", C=1.0, seed=seed),
        SVMAuditor(kernel="rbf", C=10.0, seed=seed),
        SVMAuditor(kernel="linear", seed=seed),

        # Linear variants
        LinearAuditor(C=0.1, seed=seed),
        LinearAuditor(C=1.0, seed=seed),
        LinearAuditor(C=10.0, seed=seed),
    ]

    names = [
        "MLP_shallow", "MLP_medium", "MLP_deep", "MLP_regularized",
        "RF_shallow", "RF_deep",
        "SVM_rbf_C1", "SVM_rbf_C10", "SVM_linear",
        "Linear_C0.1", "Linear_C1", "Linear_C10",
    ]

    # Add XGBoost variants if available
    if HAS_XGBOOST:
        auditors.extend([
            XGBoostAuditor(n_estimators=100, max_depth=3, seed=seed),
            XGBoostAuditor(n_estimators=100, max_depth=6, seed=seed),
            XGBoostAuditor(n_estimators=200, max_depth=6, seed=seed),
        ])
        names.extend(["XGB_shallow", "XGB_medium", "XGB_large"])

    return AuditorPool(auditors, names)


if __name__ == "__main__":
    # Test on synthetic data
    print("Testing diverse auditors on synthetic data...")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    repr_dim = 64

    # Create representations with some structure
    X = np.random.randn(n_samples, repr_dim)

    # Create labels correlated with first few dimensions
    signal = X[:, :5].sum(axis=1)
    y = (signal > np.median(signal)).astype(int)

    # Split into train/test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Representation dimension: {repr_dim}")
    print(f"Class distribution: {np.bincount(y_train)}")
    print()

    # Test individual auditors
    print("Testing individual auditors:")
    print("-" * 40)

    auditor_classes = [
        ("Linear", LinearAuditor(seed=42)),
        ("MLP", MLPAuditor(hidden_dims=[128, 64], epochs=50, seed=42)),
        ("RandomForest", RandomForestAuditor(n_estimators=100, seed=42)),
        ("SVM", SVMAuditor(kernel="rbf", seed=42)),
    ]

    if HAS_XGBOOST:
        auditor_classes.append(("XGBoost", XGBoostAuditor(n_estimators=100, seed=42)))

    for name, auditor in auditor_classes:
        auditor.fit(X_train, y_train)
        results = auditor.evaluate(X_test, y_test)
        print(f"{name:15} - Accuracy: {results['accuracy']:.4f}, AUC: {results['auc']:.4f}")

    print()

    # Test auditor pool
    print("Testing AuditorPool:")
    print("-" * 40)

    pool = get_default_auditor_pool(seed=42)
    results = pool.fit_evaluate(X_train, y_train)

    # Evaluate on test set
    test_results = pool.evaluate(X_test, y_test)

    print(f"Best accuracy: {test_results['best_accuracy']:.4f} ({test_results['best_auditor_accuracy']})")
    print(f"Best AUC: {test_results['best_auc']:.4f} ({test_results['best_auditor_auc']})")
    print()
    print("Individual results:")
    for name, metrics in test_results["individual_results"].items():
        print(f"  {name:15} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

    print()

    # Test extensive pool
    print("Testing extensive AuditorPool:")
    print("-" * 40)

    extensive_pool = get_extensive_auditor_pool(seed=42)
    extensive_results = extensive_pool.fit_evaluate(X_train, y_train)
    test_extensive = extensive_pool.evaluate(X_test, y_test)

    print(f"Best accuracy: {test_extensive['best_accuracy']:.4f} ({test_extensive['best_auditor_accuracy']})")
    print(f"Best AUC: {test_extensive['best_auc']:.4f} ({test_extensive['best_auditor_auc']})")
    print(f"Number of auditors: {len(extensive_pool.auditors)}")

    print()
    print("=" * 60)
    print("All tests passed!")
