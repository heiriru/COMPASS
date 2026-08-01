"""Filesystem layout for all benchmark artifacts."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkPaths:
    root: Path

    @classmethod
    def default(cls):
        return cls(Path(__file__).resolve().parent / "artifacts")

    def ensure(self):
        for path in (
            self.data, self.checkpoints, self.posteriors, self.metrics,
            self.figures, self.manifests, self.tables, self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def data(self): return self.root / "data"
    @property
    def checkpoints(self): return self.root / "checkpoints"
    @property
    def posteriors(self): return self.root / "posteriors"
    @property
    def metrics(self): return self.root / "metrics"
    @property
    def figures(self): return self.root / "figures"
    @property
    def manifests(self): return self.root / "manifests"
    @property
    def tables(self): return self.root / "tables"
    @property
    def logs(self): return self.root / "logs"
