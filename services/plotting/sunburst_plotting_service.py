from collections import deque
from typing import Any, Sequence
from plotly import graph_objects


class SunburstPlottingService:
    def __init__(
        self,
        default_width: int | None = None,
        default_height: int | None = None,
    ) -> None:
        self.default_width = default_width
        self.default_height = default_height

    def get_sunburst_data(
        self, hierarchy_tree: dict[str, Any], max_depth: int | None = None
    ) -> dict[str, list[Any]]:
        values = {"labels": [], "parents": [], "sizes": [], "items": []}

        stack = deque([(hierarchy_tree, 0, None)])

        while stack:
            current_node, depth, parent_name = stack.pop()

            name = current_node.get("name", "")
            items = current_node.get("items", [])
            size = len(items)

            if name != "" and (max_depth is None or depth <= max_depth):
                values["labels"].append(name)
                values["parents"].append(parent_name if parent_name != "" else None)
                values["sizes"].append(size)
                values["items"].append(items)

            stack.extend(
                (child, depth + 1, name) for child in current_node.get("children", [])
            )
        return values

    def plot_sunburst(
        self,
        sunburst_data: dict[str, Any] | None = None,
        hierarchy_tree: dict[str, Any] | None = None,
        label_texts: Sequence[str] | None = None,
        max_depth: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        if sunburst_data is None and hierarchy_tree is None:
            raise Exception("Both sunburst_data and hierarchy_tree cannot be None")
        if sunburst_data is None:
            sunburst_data = self.get_sunburst_data(hierarchy_tree, max_depth=max_depth)

        if width is None:
            width = self.default_width
        if height is None:
            height = self.default_height

        fig_sunburst = graph_objects.Figure(
            graph_objects.Sunburst(
                labels=sunburst_data["labels"],
                parents=sunburst_data["parents"],
                values=sunburst_data["sizes"],
                text=label_texts,
                maxdepth=max_depth,
            )
        )
        fig_sunburst.update_layout(
            width=width,
            height=height,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        fig_sunburst.show()
