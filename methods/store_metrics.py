"""Store all metrics in a file."""

from typing import Callable

from data.report import Report


def store_metrics(
    write_summary_report: Callable[[Report], None],
) -> None:
    def store(
        report: Report,
    ) -> None:
        return write_summary_report(report)

    return store