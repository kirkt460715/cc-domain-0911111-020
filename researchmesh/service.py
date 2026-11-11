import inspect
from .store import Store
from .normalization.changes import ChangeFeed
from .snapshots.catalogue import PublishedCatalogue
from .normalization.dictionary import Dictionary
from .normalization.text import Normalizer
from .normalization.ingestion import Sources
from .linkage.evidence import Evidence
from .linkage.candidates import Candidates
from .linkage.quality import Quality
from .decisions.graph import DecisionGraph
from .decisions.review import ReviewQueue
from .snapshots.master import MasterRecords
from .snapshots.releases import Releases
from .calibration.labels import LabelLedger
from .calibration.calibration import Calibration
from .contracts import fields, fail


class ResearchMesh:
    def __init__(self, path=":memory:"):
        self.store = Store(path)
        self.dictionary = Dictionary(self.store)
        self.normalizer = Normalizer(self.dictionary)
        self.sources = Sources(self.store, self.normalizer)
        self.evidence = Evidence(self.sources)
        self.candidates = Candidates(self.store, self.sources, self.evidence)
        self.graph = DecisionGraph(self.store, self.sources, self.evidence)
        self.sources.on_change = self.graph.rebuild
        self.changes = ChangeFeed(self.store, self.sources)
        self.review = ReviewQueue(self.store, self.evidence, self.graph)
        self.labels = LabelLedger(self.store, self.evidence)
        self.calibration = Calibration(
            self.store, self.labels, self.evidence, self.sources, self.review
        )
        self.masters = MasterRecords(
            self.store, self.sources, self.normalizer, self.graph
        )
        self.releases = Releases(self.store, self.sources, self.graph, self.masters)
        self.catalogue = PublishedCatalogue(self.releases)
        self.quality = Quality(
            self.store,
            self.sources,
            self.normalizer,
            self.candidates,
            self.graph,
            self.masters,
        )

    def actions(self):
        definitions = [
            (
                "changes",
                self.changes,
                ["inspect", "apply", "history", "compare_generations"],
            ),
            (
                "catalogue",
                self.catalogue,
                ["search", "external", "source_record", "csv"],
            ),
            (
                "source",
                self.sources,
                [
                    "register",
                    "get",
                    "priority",
                    "record",
                    "list_records",
                    "snapshot",
                    "reindex",
                    "history",
                    "parse",
                ],
            ),
            (
                "dictionary",
                self.dictionary,
                ["put", "remove", "list", "impact", "resolve"],
            ),
            ("evidence", self.evidence, ["compare", "explain"]),
            (
                "candidate",
                self.candidates,
                ["for_record", "batch", "index_stats", "coverage"],
            ),
            (
                "graph",
                self.graph,
                [
                    "preview",
                    "decide",
                    "decision",
                    "revoke",
                    "cluster",
                    "members",
                    "clusters",
                    "path",
                    "impact",
                    "verify",
                    "rebuild",
                ],
            ),
            (
                "review",
                self.review,
                [
                    "enqueue",
                    "get",
                    "claim",
                    "refresh",
                    "resolve",
                    "dismiss",
                    "release",
                    "pending",
                    "stats",
                ],
            ),
            (
                "master",
                self.masters,
                [
                    "build",
                    "all",
                    "pin",
                    "unpin",
                    "conflicts",
                    "orphan_pins",
                    "discard_orphan_pin",
                ],
            ),
            (
                "release",
                self.releases,
                [
                    "preview",
                    "freeze",
                    "get",
                    "diff",
                    "export",
                    "verify",
                    "list",
                    "lineage",
                ],
            ),
            (
                "quality",
                self.quality,
                [
                    "source_profile",
                    "cluster_profile",
                    "labeled_pairs",
                    "threshold_sweep",
                    "candidate_recall",
                    "identifier_collisions",
                    "verify",
                ],
            ),
            (
                "label",
                self.labels,
                ["record", "withdraw", "get", "pair_view", "history", "conflicts", "stats"],
            ),
            (
                "calibration",
                self.calibration,
                [
                    "freeze_dataset",
                    "get_freeze",
                    "run_validation",
                    "evaluate_holdout",
                    "get_experiment",
                    "list_experiments",
                    "suggest_batch",
                    "list_suggestions",
                ],
            ),
            ("storage", self.store, ["verify_audit", "audit_page", "backup"]),
        ]
        result = {}
        for prefix, component, methods in definitions:
            for method in methods:
                result[prefix + "." + method] = getattr(component, method)
        return result

    def dispatch(self, request):
        fields(request, {"action", "arguments"}, {"action"})
        actions = self.actions()
        action = request["action"]
        if not isinstance(action, str) or action not in actions:
            fail("action", "unknown research-mesh operation")
        arguments = request.get("arguments", {})
        if not isinstance(arguments, dict):
            fail("input", "operation arguments must be an object")
        method = actions[action]
        try:
            inspect.signature(method).bind(**arguments)
        except TypeError as exc:
            fail("input", "operation arguments differ", reason=str(exc))
        return method(**arguments)

    def describe(self):
        result = []
        for action, method in sorted(self.actions().items()):
            parameters = []
            for parameter in inspect.signature(method).parameters.values():
                value = {
                    "name": parameter.name,
                    "required": parameter.default is inspect.Parameter.empty,
                }
                if parameter.default is not inspect.Parameter.empty:
                    value["default"] = parameter.default
                parameters.append(value)
            result.append({"action": action, "parameters": parameters})
        return result

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
