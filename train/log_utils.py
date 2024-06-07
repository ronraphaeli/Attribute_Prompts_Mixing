from typing import Optional

from rich.console import Console, JustifyMethod
from rich.filesize import pick_unit_and_suffix
from rich.highlighter import Highlighter
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.style import StyleType
from rich.table import Column
from rich.text import Text
from rich.theme import Theme


class SpeedColumn(TextColumn):
    _renderable_cache = {}

    def __init__(
        self,
        style: StyleType = "none",
        justify: JustifyMethod = "left",
        markup: bool = True,
        highlighter: Optional[Highlighter] = None,
        table_column: Optional[Column] = None,
    ) -> None:
        super().__init__("", style, justify, markup, highlighter, table_column)

    def render(self, task: Task) -> Text:
        self.text_format = f"{task.speed:.3f} steps/s" if task.speed else "Nan steps/s"
        return super().render(task)


class SpeedColumn1(TaskProgressColumn):
    _renderable_cache = {}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def render_speed(cls, speed: Optional[float]) -> Text:
        """Render the speed in iterations per second.

        Args:
            task (Task): A Task object.

        Returns:
            Text: Text object containing the task speed.
        """
        if speed is None:
            return Text("", style="progress.speed")
        unit, suffix = pick_unit_and_suffix(
            int(speed),
            ["", "×10³", "×10⁶", "×10⁹", "×10¹²"],
            1000,
        )
        data_speed = speed / unit
        return Text(f"{data_speed:.2f}{suffix} steps/s", style="progress.speed")

    def render(self, task: "Task") -> Text:
        return self.render_speed(task.finished_speed or task.speed)


def get_console():
    console = Console(
        theme=Theme(
            {
                "progress.elapsed": "bright_blue",
                "progress.remaining": "blue",
                "progress.percentage": "green",
                "progress.speed": "magenta",
            }
        )
    )
    return console


def config_log(console, logger):
    console_handler = RichHandler(console=console, rich_tracebacks=True)
    logger.addHandler(console_handler)


def get_progress(console):
    progress = Progress(
        SpinnerColumn("arc"),
        TextColumn(
            "[bold blue]{task.description} \[epoch {task.fields[epoch]}]",
            justify="right",
        ),
        BarColumn(bar_width=None),
        "{task.completed}/{task.total} steps",
        "([progress.percentage]{task.percentage:>3.1f}%)",
        "•",
        SpeedColumn1(justify="right"),
        "•",
        "[yellow]{task.fields[loss]:.3e}",
        "•",
        TimeElapsedColumn(),
        "/",
        TimeRemainingColumn(elapsed_when_finished=True),
        console=console,
        refresh_per_second=1,
    )
    return progress
