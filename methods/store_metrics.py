"""Store all metrics in a file."""

from typing import Callable

from data.report import Report


def store_metrics(
    write_summary_report: Callable[[Report], None],
    store_report_to_file: Callable[[Report], None],
) -> None:
    def store(
        report: Report,
    ) -> None:
        write_summary_report(report)
        return store_report_to_file(report)
        

    return store