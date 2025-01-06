import datetime
import os
import pickle
from typing import Optional
import matplotlib.pyplot as plt
import cv2  # type: ignore[import]

from dataclasses import dataclass

from base.helpers import FIGURES_FOLDER, SLOTS_PER_EPOCH, Status
from base.statistics import EpochEntityStat
from data.file_manager import FileManager


def date_to_slots(date: str) -> int:
    datetime_object = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    timestamp = int(datetime_object.timestamp())
    return (timestamp - 1704879491) // 12 + 8171589


@dataclass
class Label:
    label: str
    shown: bool


def plot_missed(
    title: str,
    labels: list[Label],
    reorged: list[int],
    missed: list[int],
    out_path: Optional[str] = None,
):
    dates: list[str] = [label.label for label in labels]

    blue_densities = reorged
    orange_densities = missed

    plt.bar(dates, blue_densities, color="blue")

    bottom = blue_densities
    plt.bar(dates, orange_densities, bottom=bottom, color="orange")

    plt.xlabel("Date")
    plt.ylabel("Density")
    plt.title("Timestamp Density Plot")
    plt.xticks(rotation=45)
    plt.ylim(0, 152)
    plt.legend(["Reorged", "Missed"])
    plt.xticks([i for i, label in enumerate(labels) if label.shown])

    plt.title(title)
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
        plt.close()


def create_video_from_images(filenames: list[str], video_path: str) -> None:
    """
    Creates a video from a list of image filenames.

    Parameters:
        filenames (list[str]): List of image filenames.
        video_path (str): Path to save the output video.
    """
    # Read the first image to get dimensions
    first_image = cv2.imread(filenames[0])
    height, width, _ = first_image.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for the output video
    video_writer = cv2.VideoWriter(video_path, fourcc, 2, (width, height))

    # Loop through each image and write it to the video
    for filename in filenames:
        image = cv2.imread(filename)
        video_writer.write(image)

    # Release the VideoWriter object
    video_writer.release()

    print(f"Video saved as {video_path}")


class Plotter:
    def __init__(self, reorged_slots: list[int], missed_slots: list[int]):
        self.reorged = reorged_slots
        self.missed_slots = missed_slots

    def plot(
        self, start_slot: int, end_slot: int, title: str, out_path: Optional[str] = None
    ):
        reorged_slots = [
            slot for slot in self.reorged if start_slot <= slot <= end_slot
        ]
        missed_slots = [
            slot for slot in self.missed_slots if start_slot <= slot <= end_slot
        ]

        reorged_clusters = [0] * 32
        missed_clusters = [0] * 32

        for slot in reorged_slots:
            reorged_clusters[slot % 32] += 1
        for slot in missed_slots:
            missed_clusters[slot % 32] += 1

        labels = [Label(label=str(i), shown=True) for i in range(32)]

        plot_missed(
            title=title,
            labels=labels,
            reorged=reorged_clusters,
            missed=missed_clusters,
            out_path=out_path,
        )


class SequencePlotter:
    def __init__(
        self,
        plotter: Plotter,
    ):
        self.plotter = plotter

    def start_plotting(
        self,
        out_folder: str,
        start_date: datetime.datetime,
        consecutive_days: int,
        end_date: datetime.datetime,
        jump_units: int,
        generate_video=True,
    ):
        i = 0
        filenames = []
        while start_date + datetime.timedelta(days=consecutive_days - 1) <= end_date:
            start_slot = date_to_slots(
                date=f"{start_date.year}-{start_date.month}-{start_date.day} 0:0:0"
            )
            ending_date = start_date + datetime.timedelta(days=consecutive_days - 1)
            end_slot = date_to_slots(
                date=f"{ending_date.year}-{ending_date.month}-{ending_date.day} 0:0:0"
            )
            start_str = start_date.strftime("%Y %b %d")
            end_str = ending_date.strftime("%Y %b %d")
            title = f"{start_str} - {end_str}"
            out_path = os.path.join(out_folder, f"{i}.png")
            filenames.append(out_path)
            self.plotter.plot(
                start_slot=start_slot, end_slot=end_slot, title=title, out_path=out_path
            )

            i += 1
            start_date = start_date + datetime.timedelta(days=jump_units)
        if generate_video:
            start_str = start_date.strftime("%Y.%m.%d")
            end_str = end_date.strftime("%Y.%m.%d")
            create_video_from_images(
                filenames=filenames,
                video_path=os.path.join(FIGURES_FOLDER, f"{start_str}-{end_str}.mp4"),
            )


def plot_missed_reorged_stats():
    beaconchain = FileManager.file_manager().beaconchain()
    reorged_slots = [
        slot for slot, block in beaconchain.items() if block.status == Status.REORGED
    ]
    missed_slots = [
        slot for slot, block in beaconchain.items() if block.status == Status.MISSED
    ]
    plotter = Plotter(reorged_slots=reorged_slots, missed_slots=missed_slots)
    seq_plotter = SequencePlotter(plotter)
    folder = os.path.join(FIGURES_FOLDER, "frames")
    os.makedirs(folder, exist_ok=True)
    seq_plotter.start_plotting(
        out_folder=folder,
        start_date=datetime.datetime(year=2022, month=9, day=6),
        consecutive_days=14,
        end_date=datetime.datetime(year=2024, month=6, day=10),
        jump_units=7,
    )


def plot_lattices_2graph(
    expected_values: dict[int, float],
    actual_values: dict[int, float],
    entity: str,
    out_folder: str = FIGURES_FOLDER,
):
    os.makedirs(out_folder, exist_ok=True)
    # Extract points from the dictionaries
    x1, y1 = zip(*expected_values.items())
    x2, y2 = zip(*actual_values.items())

    # Create a new figure
    plt.figure()

    # Plot the first set of points and connect them with lines
    plt.plot(x1, y1, "o-", color="blue", label="Expected number of proposed blocks")

    # Plot the second set of points and connect them with lines
    plt.plot(x2, y2, "o-", color="red", label="Average of proposed blocks")

    # Adding grid lines with a grating interval of 0.5 units
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.minorticks_on()
    plt.gca().xaxis.set_major_locator(plt.AutoLocator())
    plt.gca().yaxis.set_minor_locator(plt.AutoLocator())

    # Adding labels and title
    plt.xlabel("Epoch")
    plt.ylabel("Proposed Blocks")

    out_path = os.path.join(out_folder, f"{entity}.png")
    entity_show = entity if len(entity) <= 12 else f"{entity[:10]}..."

    plt.title(f"{entity_show} - Proposed blocks")
    plt.legend()

    plt.savefig(out_path)
    plt.close()


def plot_expected_average(jumps: int = 1575 * 2):
    with open("data/processed_data/proposed_blocks_stats.pkl", "rb") as f:
        data: dict[int, EpochEntityStat] = pickle.load(f)

    min_epoch = min(data)
    max_epoch = max(data)

    epochs = list(range(min_epoch - 1 + jumps // 2, max_epoch, jumps))
    first = epochs[0]

    entities = list(data.items())[0][1].actual.keys()
    for entity in entities:
        epoch_to_avg_list: dict[int, list[float]] = {epoch: [] for epoch in epochs}
        epoch_to_exp_list: dict[int, list[float]] = {epoch: [] for epoch in epochs}

        for epoch, epoch_entity_stat in data.items():
            cluster = round((epoch - first) / jumps) * jumps + first
            if cluster not in epoch_to_avg_list:
                epoch_to_avg_list[cluster] = []
                epoch_to_exp_list[cluster] = []
            epoch_to_avg_list[cluster].append(epoch_entity_stat.actual[entity])
            epoch_to_exp_list[cluster].append(
                epoch_entity_stat.expected[entity] * SLOTS_PER_EPOCH
            )

        epoch_to_avg = {
            epoch: (sum(collection) / len(collection) if len(collection) else 0)
            for epoch, collection in epoch_to_avg_list.items()
        }
        epoch_to_exp = {
            epoch: (sum(collection) / len(collection) if len(collection) else 0)
            for epoch, collection in epoch_to_exp_list.items()
        }

        plot_lattices_2graph(
            expected_values=epoch_to_exp,
            actual_values=epoch_to_avg,
            entity=entity,
            out_folder=os.path.join(FIGURES_FOLDER, "entity_stats"),
        )


if __name__ == "__main__":
    plot_expected_average()  # plot_missed_reorged_stats()
