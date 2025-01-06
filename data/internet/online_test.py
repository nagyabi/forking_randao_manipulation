from base.beaconchain import BlockData, Validator
from base.helpers import DEFAULT, GREEN, MISSING, RED, Status
import data.internet

import data.internet.connection
import data.internet.beaconscan.scrape


def scrape_test() -> dict[str, bool]:
    """
    Testing connection to websites and correctness of scraping functions

    Returns:
        dict[str, bool]: mapping from service to whether its online(/offline=False)
    """

    functions_with_args = [
        (
            "beaconscan.scrape.slot_data",
            data.internet.beaconscan.scrape.slot_data,
            [8530623],
        ),
        (
            "beaconscan.scrape.validators",
            data.internet.beaconscan.scrape.validators,
            [1, 10],
        ),
    ]

    expected = [
        BlockData(
            slot=8530623,
            proposer_index=MISSING,
            block=MISSING,
            parent_block_slot=MISSING,
            status=Status.REORGED,
            randao_reveal="0x869166b5304251f8cad0261d426b9642cd603bf70769042ee26a168e7450afc6f658ec0f83505d3d11fcb0bddd619667133afe491122b99ffd5dd5b2cd90435016d3e0eec9a7cf2082d88a548fb4fc0e6943c98807a4f5b6f93581d8aacf95e6",
            randomness=MISSING,
            block_root="0x271808b7f19d6d6daf351530965df64c99ba8619e9fbed7c8d7d9a72fbd99a57",
            parent_root="0x992cea3b7f799d3f9a7bf8fb713df48c1210f14839bca6a873c8e1f8bf55ca0b",
        ),
        {
            0: Validator(
                index=0,
                effective_balance=32,
                slashed=False,
                public_key="0x933ad9491b62059dd065b560d256d8957a8c402cc6e8d8ee7290ae11e8f7329267a8811c397529dac52ae1342ba58c95",
                start_slot=0,
                end_slot=None,
            ),
            1: Validator(
                index=1,
                effective_balance=32,
                slashed=False,
                public_key="0xa1d1ad0714035353258038e964ae9675dc0252ee22cea896825c01458e1807bfad2f9969338798548d9858a571f7425c",
                start_slot=0,
                end_slot=None,
            ),
            2: Validator(
                index=2,
                effective_balance=32,
                slashed=False,
                public_key="0xb2ff4716ed345b05dd1dfc6a5a9fa70856d8c75dcc9e881dd2f766d5f891326f0d10e96f3a444ce6c912b69c22c6754d",
                start_slot=0,
                end_slot=None,
            ),
            3: Validator(
                index=3,
                effective_balance=32,
                slashed=False,
                public_key="0x8e323fd501233cd4d1b9d63d74076a38de50f2f584b001a5ac2412e4e46adb26d2fb2a6041e7e8c57cd4df0916729219",
                start_slot=0,
                end_slot=None,
            ),
            4: Validator(
                index=4,
                effective_balance=32,
                slashed=False,
                public_key="0xa62420543ceef8d77e065c70da15f7b731e56db5457571c465f025e032bbcd263a0990c8749b4ca6ff20d77004454b51",
                start_slot=0,
                end_slot=None,
            ),
            5: Validator(
                index=5,
                effective_balance=32,
                slashed=False,
                public_key="0xb2ce0f79f90e7b3a113ca5783c65756f96c4b4673c2b5c1eb4efc2228025944106d601211e8866dc5b50dc48a244dd7c",
                start_slot=0,
                end_slot=None,
            ),
            6: Validator(
                index=6,
                effective_balance=32,
                slashed=False,
                public_key="0xa16c530143fc72497a85e0de237be174f773cc1e496a94bd13d02708e0fdc1b5c7d25a9c2c05f09d5de8b8ed2bf8e0d2",
                start_slot=0,
                end_slot=None,
            ),
            7: Validator(
                index=7,
                effective_balance=32,
                slashed=False,
                public_key="0xa25da1827014cd3bc6e7b70f1375750935a16f00fbe186cc477c204d330cac7ee060b68587c5cdcfae937176a4dd2962",
                start_slot=0,
                end_slot=None,
            ),
            8: Validator(
                index=8,
                effective_balance=32,
                slashed=False,
                public_key="0x8078c7f4ab6f9eaaf59332b745be8834434af4ab3c741899abcff93563544d2e5a89acf2bec1eda2535610f253f73ee6",
                start_slot=0,
                end_slot=None,
            ),
            9: Validator(
                index=9,
                effective_balance=32,
                slashed=False,
                public_key="0xb016e31f633a21fbe42a015152399361184f1e2c0803d89823c224994af74a561c4ad8cfc94b18781d589d03e952cd5b",
                start_slot=0,
                end_slot=None,
            ),
        },
    ]
    result: dict[str, int] = {}
    for (name, func, arg), expec in zip(functions_with_args, expected):
        try:
            actual = func(*arg)
        except Exception:
            actual = None
        result[name] = actual == expec
    return result


def print_test_results(
    service_to_online_dict: dict[str, bool], should_upper: bool = True
):
    for name, is_online in service_to_online_dict.items():
        service_status = "ONLINE" if is_online else "OFFLINE"
        color = GREEN if service_status == "ONLINE" else RED
        if should_upper:
            print(f"{name.upper()}: {color}{service_status}{DEFAULT}")
        else:
            print(f"{name}: {color}{service_status}{DEFAULT}")


if __name__ == "__main__":
    service_to_is_online = scrape_test()
    print_test_results(service_to_is_online)
