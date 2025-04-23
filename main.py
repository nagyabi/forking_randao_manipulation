import argparse
import json
import os

import numpy as np

from theory.method.solution import NewMethodSolver
from theory.method.quant.run_user_quant import run_quant_model
from base.helpers import CACHE_FOLDER, SLOTS_PER_EPOCH, read_api_keys
from base.statistics import index_to_entity_job
from data.collect.beaconchain_grinder import grind_beaconchain
from data.collect.data_all_collection import data_full
from data.collect.validator_grinder import (
    address_grinding,
    deposit_grinding,
    validator_grinding,
)
from data.file_manager import FileManager
from data.internet.beaconcha.api import BeaconChaAPITester
from data.internet.etherscan.api import EtherscanAPITester
from data.internet.online_test import print_test_results, scrape_test
from data.internet.utils import APITester
from data.process.make_alternatives import produce_needed_alternatives
from data.process.statistics import make_statistics
from theory.selfish_mixing import selfish_mixing


def format_cache_path(filename: str) -> str:
    splitted = filename.split("-")
    splitted[0] = splitted[0].replace("_", ".")
    return " ".join(splitted)


def frange(start, stop, step):
    while start <= stop:
        yield round(start, 10)  # Using round to avoid floating-point precision issues
        start += step


def parse_alphas(alphas_str: str) -> list[np.float64]:
    num_strings = [text.strip() for text in alphas_str.split(",")]
    try:
        alphas: list[np.float64] = []
        for part in num_strings:
            if ":" in part:
                start, stop, step = [np.float64(p) for p in part.split(":")]
                alphas.extend(frange(start, stop, step))
            else:
                alphas.append(np.float64(part))
        return alphas

    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid value(s) provided. Please provide either a single float or a comma-separated list of floats."
        )


def parse_keys(keys: str) -> list[str]:
    return [key.strip() for key in keys.split(",")]


def show_cached(cache_folder: str) -> None:
    cached_folders: list[str] = []
    if os.path.exists(cache_folder):
        entries = os.listdir(cache_folder)
        cached_folders = [
            entry
            for entry in entries
            if os.path.isdir(os.path.join(cache_folder, entry))
        ]
    if len(cached_folders) == 0:
        print("No cached records found")
    else:
        formatted_cache_records = [
            format_cache_path(filename) for filename in cached_folders
        ]
        for record in formatted_cache_records:
            print(record)


def prerequisites(file_to_message: dict[str, str]) -> bool:
    all_done = True
    for file, message in file_to_message.items():
        if not os.path.exists(file):
            all_done = False
            print(message)
    return all_done


def main():
    parser = argparse.ArgumentParser(
        description="Example run: main.py --theory --alphas 0.151,0.181 --size-postfix 7"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config-api-keys",
        metavar="CONFIG-API-KEYS",
        choices=["beaconcha", "etherscan"],
        help="Specify which beaconchain explorers you would like to configure: (beaconcha, etherscan)",
    )
    group.add_argument(
        "--data",
        metavar="DATA",
        choices=[
            "test-scrape",
            "beaconchain",
            "validators",
            "deposits",
            "address-txn",
            "index-to-entity",
            "full",
        ],
        help="Specify which kind of data you want",
    )
    group.add_argument(
        "--theory",
        metavar="THEORY",
        nargs="?",
        const="Theory: The final distribution of RO is calculated along with indicators of possible attacks",
    )
    group.add_argument(
        "--alternatives",
        metavar="ALTERNATIVES",
        nargs="?",
        const="Proposer orderings are calculated to the  corresponding alternative BeaconStates",
    )
    group.add_argument(
        "--show-cached",
        metavar="SHOW-CACHED",
        nargs="?",
        const="Show precalculated indicators of theoretical attacks",
    )
    group.add_argument(
        "--statistics",
        metavar="STATISTICS",
        nargs="?",
        const="Making statistics from the data and from the theory's output",
    )

    # API keys
    parser.add_argument(
        "--action", choices=["help", "overwrite", "append", "test", "filter", "delete"]
    )
    parser.add_argument(
        "--key-values",
        type=parse_keys,
        help="Specify API keys for chosen website (required for owerwrite and append)",
    )
    parser.add_argument(
        "--test-values",
        action="store_true",
        help="The specified API keys will be tested",
    )

    # Data
    # Beaconchain
    parser.add_argument(
        "--max-epoch",
        type=int,
        help="Specify the maximum epoch for beaconchain data collection",
    )
    parser.add_argument(
        "--min-epoch",
        type=int,
        help="Specify the minimum epoch for beaconchain data collection",
    )

    # Validators
    parser.add_argument(
        "--max-validator",
        type=int,
        help="(Optional flag) Specify the maximum proposer index for the validator data collection",
    )

    # Deposits
    parser.add_argument(
        "--max-block",
        type=int,
        help="Specify the maximum block for the deposit collection",
    )

    # Full data & Validators
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Collecting validators will start where it crashed/stopped",
    )

    # Markov chain & Statistics & Alternatives

    parser.add_argument(
        "--size-postfix",
        type=int,
        default=6,
        help="A single integer of how much slots before the epoch boundary will count in the attack string",
    )
    parser.add_argument(
        "--size-prefix",
        type=int,
        default=2,
        help="A single integer of how much slots after the epoch boundary will count in the attack string",
    )
    # Only for theory
    parser.add_argument(
        "--alphas",
        type=parse_alphas,
        help="A single float or a comma-separated list of floats which represents the stakes",
    )
    parser.add_argument(
        "--heur-max",
        type=int,
        default=SLOTS_PER_EPOCH,
        help="Maximum of tail slot in selfish mixing theory",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="A single integer of how much iteration will be used for finetuning the value function",
    )
    parser.add_argument(
        "--selfish-mixing",
        action="store_true",
        help="When used, only the selfish mixing will be calculated",
    )
    parser.add_argument(
        "--quant", action="store_true", help="When used, a quantized model is produced"
    )
    parser.add_argument(
        "--markov-chain",
        action="store_true",
        help="When used the model is evaluated with rich eval in the end",
    )
    parser.add_argument(
        "--try-quantized",
        action="store_true",
        help="Using quantized model with provided parameters",
    )

    # Statistics
    parser.add_argument(
        "--export-folder",
        type=str,
        default="data/processed_data/delivery/",
        help="Folder of the exported file",
    )
    parser.add_argument(
        "--minimal-occurrence",
        type=int,
        default=2,
        help="Integer denoting the minimal amount of blocks for each entity to be candidates of an attack. Under this the entity won't be shown in Table 2",
    )
    parser.add_argument(
        "--only-summarized",
        action="store_true",
        help="Only the summarizing stats will be show, casewise reports will be skipped",
    )

    args = parser.parse_args()

    if args.config_api_keys:

        api_tester: APITester
        if args.config_api_keys == "beaconcha":
            api_tester = BeaconChaAPITester()
        else:
            api_tester = EtherscanAPITester()
        path = os.path.join("data/internet/apikeys", f"{args.config_api_keys}.json")
        if args.action is None:
            parser.error(
                "--action [overwrite | append | test | filter | delete] is required when using --config-api-keys"
            )
        if args.action == "help":
            if args.config_api_keys == "beaconcha":
                print(
                    "You can get beaconcha.in API keys here: https://beaconcha.in/pricing"
                )
            elif args.config_api_keys == "etherscan":
                print(
                    "You can get etherscan API keys here: https://etherscan.io/apis#pricing-plan"
                )
        elif args.action in ["overwrite", "append"]:
            if args.key_values is None:
                parser.error(
                    f"--key-values is required when using --config-api-keys {args.config_api_keys} --action {args.action}"
                )
            actual_api_keys = read_api_keys(path)
            if args.test_values:
                service_to_is_online = api_tester.test_api_keys(
                    api_keys=args.key_values
                )
                print_test_results(
                    service_to_online_dict=service_to_is_online, should_upper=False
                )
                filtered_keys = [
                    key for key, is_online in service_to_is_online.items() if is_online
                ]
                if args.action == "overwrite":
                    actual_api_keys = filtered_keys
                else:
                    actual_api_keys.extend(filtered_keys)
            else:
                if args.action == "overwrite":
                    actual_api_keys = args.key_values
                else:
                    actual_api_keys.extend(args.key_values)
            actual_api_keys = list(set(actual_api_keys))
            with open(path, "w") as f:
                json.dump(actual_api_keys, f)
        elif args.action in ["test", "filter"]:
            actual_api_keys = read_api_keys(path)
            service_to_is_online = api_tester.test_api_keys(api_keys=actual_api_keys)
            print_test_results(
                service_to_online_dict=service_to_is_online, should_upper=False
            )
            if args.action == "filter":
                filtered_keys = [
                    key for key, is_online in service_to_is_online.items() if is_online
                ]
                with open(path, "w") as f:
                    json.dump(filtered_keys, f)
        elif args.action == "delete":
            actual_api_keys = read_api_keys(path)
            for key in actual_api_keys:
                print(f"Deleting {key}..")
            with open(path, "w") as f:
                json.dump([], f)

    elif args.data:
        if args.data == "test-scrape":
            service_to_is_online = scrape_test()
            print_test_results(service_to_is_online)
            headers_path_to_link = {
                "data/internet/headers/beaconscan.header": "https://beaconscan.com/validators",
                "data/internet/headers/sid.txt": "https://beaconscan.com/validators",
            }
            for header_path, link in headers_path_to_link.items():
                if not os.path.exists(header_path):
                    print(f"Warning: No header file/session id in {header_path}. Copy the correct information by visiting this link: {link}")
        elif args.data == "beaconchain":
            if args.max_epoch is None:
                parser.error("--max-epoch is required when choosing --data beaconchain")
            grind_beaconchain(max_epoch=args.max_epoch, min_epoch=args.min_epoch)
        elif args.data == "validators":
            validator_grinding(
                regenerate=not args.resume, max_validator=args.max_validator
            )
        elif args.data == "deposits":
            if args.max_block is None:
                parser.error("--max-block is required when choosing --data deposits")
            deposit_grinding(top_block=args.max_block)
        elif args.data == "address-txn":
            if prerequisites(
                file_to_message={
                    "data/jsons/deposits.json": "First run --data deposits ..."
                }
            ):
                address_grinding()
        elif args.data == "index-to-entity":
            index_to_entity_job()
        elif args.data == "full":
            data_full(FileManager.file_manager(), regenerate_validators=not args.resume)
    elif args.alternatives:
        requests = {
            "data/jsons/beaconchain.json": "First run --data beaconchain",
            "data/jsons/validators.json": "First run --data validators ...",
            "data/jsons/deposits.json": "First run --data deposits ...",
        }
        if prerequisites(file_to_message=requests):
            produce_needed_alternatives(
                file_manager=FileManager.file_manager(),
                size_prefix=args.size_prefix,
                size_postfix=args.size_postfix,
            )
    elif args.theory:
        if args.alphas is None:
            parser.error("--alphas is required when choosing --theory")
        if not args.selfish_mixing:

            if args.try_quantized:
                if len(args.alphas) != 1:
                    parser.error(
                        f"When [--try-quantized] is used, 1 alpha value is expected, instead {len(args.alphas)} was given"
                    )
                run_quant_model(
                    alpha=args.alphas[0],
                    size_prefix=args.size_prefix,
                    size_postfix=args.size_postfix,
                    iteration=args.iterations,
                )
            else:
                for alpha in args.alphas:
                    print(f"Theoretical calculations for {alpha=}")
                    solver = NewMethodSolver(
                        alpha=alpha,
                        size_prefix=args.size_prefix,
                        size_postfix=args.size_postfix,
                    )
                    RO = solver.solve(
                        num_of_iterations=args.iterations,
                        markov_chain=args.markov_chain,
                        quant=args.quant,
                    )
                    print(f"{100 * alpha}% => {round(100 * RO / SLOTS_PER_EPOCH, 5)}%")
        else:
            selfish_mixing(
                alphas=args.alphas, iterations=args.iterations, heur_max=args.heur_max
            )
    elif args.show_cached:
        show_cached(cache_folder=CACHE_FOLDER)
    elif args.statistics:

        requests = {
            "data/jsons/beaconchain.json": "First run --data beaconchain",
            "data/jsons/validators.json": "First run --data validators ...",
            "data/jsons/deposits.json": "First run --data deposits ...",
            "data/jsons/address_txn.json": "First run --data address-txn",
            "data/processed_data/alternatives.pkl": "First run --alternatives",
            "data/jsons/entities.json": "Please provide an entity mapping",
        }
        if prerequisites(file_to_message=requests):
            make_statistics(
                file_manager=FileManager.file_manager(),
                size_prefix=args.size_prefix,
                size_postfix=args.size_postfix,
                export_folder=args.export_folder,
                minimal_occurrence=args.minimal_occurrence,
                only_summarized_stats=args.only_summarized,
            )


if __name__ == "__main__":
    main()
